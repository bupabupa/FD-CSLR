import argparse
import json
from pathlib import Path
import re
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import load_split_manifest
from scripts.collect_results import prediction_metrics


DEV_WER_PATTERN = re.compile(
    r"\bdev\s+([0-9]+(?:\.[0-9]+)?)%",
    flags=re.IGNORECASE,
)
REQUIRED_FIELDS = {"dev", "run", "config", "checkpoint"}
OPTIONAL_FIELDS = {"calibration", "priority"}


def parse_candidate(spec):
    fields = {}
    for item in spec.split(";"):
        name, separator, value = item.partition("=")
        if not separator or not name or not value:
            raise ValueError(
                "candidate fields must use key=value syntax: {}".format(spec)
            )
        if name in fields:
            raise ValueError("duplicate candidate field: {}".format(name))
        fields[name] = value
    missing = REQUIRED_FIELDS.difference(fields)
    unexpected = set(fields).difference(
        REQUIRED_FIELDS | OPTIONAL_FIELDS | {"name"}
    )
    if missing:
        raise ValueError(
            "candidate is missing fields: {}".format(sorted(missing))
        )
    if unexpected:
        raise ValueError(
            "candidate has unknown fields: {}".format(sorted(unexpected))
        )
    name = fields.get("name")
    if not name:
        raise ValueError("candidate is missing name")
    candidate = {
        "name": name,
        "dev_prediction": Path(fields["dev"]).expanduser().resolve(),
        "run_dir": Path(fields["run"]).expanduser().resolve(),
        "config": Path(fields["config"]).expanduser().resolve(),
        "checkpoint": Path(fields["checkpoint"]).expanduser().resolve(),
        "calibration": (
            Path(fields["calibration"]).expanduser().resolve()
            if fields.get("calibration")
            else None
        ),
        "priority": int(fields.get("priority", "0")),
    }
    if not candidate["dev_prediction"].is_file():
        raise ValueError(
            "Dev prediction does not exist: {}".format(
                candidate["dev_prediction"]
            )
        )
    if not candidate["run_dir"].is_dir():
        raise ValueError(
            "run directory does not exist: {}".format(candidate["run_dir"])
        )
    if not candidate["config"].is_file():
        raise ValueError(
            "config does not exist: {}".format(candidate["config"])
        )
    if not candidate["checkpoint"].is_file():
        raise ValueError(
            "checkpoint does not exist: {}".format(
                candidate["checkpoint"]
            )
        )
    return candidate


def load_optional_candidates(path):
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, list):
        raise ValueError("optional candidate file must contain a JSON list")
    candidates = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                "optional candidate {} must be an object".format(index)
            )
        allowed = REQUIRED_FIELDS | OPTIONAL_FIELDS | {"name"}
        unexpected = set(item).difference(allowed)
        if unexpected:
            raise ValueError(
                "optional candidate {} has unknown fields: {}".format(
                    index, sorted(unexpected)
                )
            )
        fields = {
            "name": item.get("name"),
            "dev": item.get("dev"),
            "run": item.get("run"),
            "config": item.get("config"),
            "checkpoint": item.get("checkpoint"),
            "calibration": item.get("calibration"),
            "priority": item.get("priority", 0),
        }
        if any(
            fields[key] in (None, "")
            for key in ("name", "dev", "run", "config", "checkpoint")
        ):
            raise ValueError(
                "optional candidate {} is missing required fields".format(
                    index
                )
            )
        parts = []
        for key in (
            "name",
            "dev",
            "run",
            "config",
            "checkpoint",
            "calibration",
            "priority",
        ):
            value = fields[key]
            if value in (None, ""):
                continue
            item_path = Path(str(value)).expanduser()
            if key != "name" and key != "priority" and not item_path.is_absolute():
                item_path = path.parent / item_path
            parts.append("{}={}".format(key, item_path))
        candidates.append(parse_candidate(";".join(parts)))
    return candidates


def authoritative_dev_wer(prediction_path, num_classes):
    metrics_path = prediction_path.parent / "dev.txt"
    if metrics_path.is_file():
        lines = metrics_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        for line in reversed(lines):
            match = DEV_WER_PATTERN.search(line)
            if match:
                return float(match.group(1)), str(metrics_path)
    metrics = prediction_metrics(prediction_path, num_classes)
    return float(metrics["wer"]), "prediction-edit-distance"


def calibration_loss(path):
    if path is None:
        return None
    with open(path, encoding="utf-8") as source:
        metrics = json.load(source)
    value = metrics.get("best_loss")
    return float(value) if value is not None else None


def existing_test_predictions(run_dir):
    return sorted(
        str(path)
        for path in run_dir.rglob("predictions-test.jsonl")
        if path.is_file()
    )


def rank_candidates(candidates, num_classes):
    output = []
    names = set()
    for candidate in candidates:
        if candidate["name"] in names:
            raise ValueError(
                "duplicate candidate name: {}".format(candidate["name"])
            )
        names.add(candidate["name"])
        wer, wer_source = authoritative_dev_wer(
            candidate["dev_prediction"], num_classes
        )
        nll = calibration_loss(candidate["calibration"])
        output.append({
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in candidate.items()
            },
            "dev_wer": wer,
            "dev_wer_source": wer_source,
            "calibration_nll": nll,
        })
    if not output:
        raise ValueError("at least one candidate is required")
    selected = min(
        output,
        key=lambda value: (
            value["dev_wer"],
            (
                value["calibration_nll"]
                if value["calibration_nll"] is not None
                else float("inf")
            ),
            value["priority"],
            value["name"],
        ),
    )
    return {
        "selection_rule": (
            "minimum authoritative D_sel Dev WER; exact ties use "
            "D_cal NLL when available, then predefined complexity priority; "
            "D_test is forbidden before this lock"
        ),
        "candidates": output,
        "selected": selected,
    }


def write_shell_environment(path, result):
    selected = result["selected"]
    values = {
        "LOCKED_NAME": selected["name"],
        "LOCKED_DEV_WER": "{:.10f}".format(selected["dev_wer"]),
        "LOCKED_CALIBRATION_NLL": (
            ""
            if selected["calibration_nll"] is None
            else "{:.10f}".format(selected["calibration_nll"])
        ),
        "LOCKED_DEV_PREDICTION": selected["dev_prediction"],
        "LOCKED_RUN_DIR": selected["run_dir"],
        "LOCKED_CONFIG": selected["config"],
        "LOCKED_CHECKPOINT": selected["checkpoint"],
        "LOCKED_PRIORITY": str(selected["priority"]),
    }
    lines = [
        "{}={}".format(name, shlex.quote(str(value)))
        for name, value in values.items()
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--num-classes", type=int, default=1296)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shell-output", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument(
        "--allow-existing-test-predictions",
        action="store_true",
    )
    args = parser.parse_args()

    candidates = [parse_candidate(spec) for spec in args.candidate]
    optional_path = (
        Path(args.output).expanduser().resolve().parent
        / "optional_candidates.json"
    )
    candidates.extend(load_optional_candidates(optional_path))
    names = [candidate["name"] for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate name across CLI and optional file")
    if not args.allow_existing_test_predictions:
        for candidate in candidates:
            existing = existing_test_predictions(candidate["run_dir"])
            if existing:
                raise ValueError(
                    "test predictions exist before candidate lock for {}: {}"
                    .format(candidate["name"], existing)
                )
    result = rank_candidates(candidates, args.num_classes)
    if args.protocol is not None:
        result["protocol"] = str(args.protocol)
    if args.split_manifest is not None:
        manifest = load_split_manifest(args.split_manifest)
        result["split_sha256"] = manifest["split_sha256"]
        result["split_manifest"] = str(
            Path(args.split_manifest).expanduser().resolve()
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")
    write_shell_environment(args.shell_output, result)
    print(json.dumps(result["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
