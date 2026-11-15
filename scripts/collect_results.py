import argparse
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_predictions import read_predictions, sentence_statistics


CHECKPOINT_PATTERN = re.compile(
    r"^dev_([0-9]+(?:\.[0-9]+)?)_epoch([0-9]+)_model\.pt$"
)
MEAN_LOSS_PATTERN = re.compile(r"Mean training loss:")
DEV_HISTORY_PATTERN = re.compile(r"Dev WER:\s*([0-9]+(?:\.[0-9]+)?)%")


def parse_run_spec(spec):
    name, separator, value = spec.partition("=")
    if not separator or not name or not value:
        raise ValueError(
            "--run must use NAME=RUN_DIR, NAME=RUN_DIR@PRED_DIR, "
            "or NAME=RUN_DIR@PRED_DIR@CALIBRATION_JSON"
        )
    parts = value.split("@")
    if len(parts) > 3 or any(not part for part in parts):
        raise ValueError("invalid run specification: {}".format(spec))
    run_value = parts[0]
    prediction_value = parts[1] if len(parts) >= 2 else None
    calibration_value = parts[2] if len(parts) == 3 else None
    return {
        "name": name,
        "run_dir": Path(run_value).expanduser().resolve(),
        "prediction_dir": (
            Path(prediction_value).expanduser().resolve()
            if prediction_value is not None
            else None
        ),
        "calibration_path": (
            Path(calibration_value).expanduser().resolve()
            if calibration_value is not None
            else None
        ),
    }


def parse_training_history(run_dir):
    log_path = run_dir / "log.txt"
    if not log_path.is_file():
        return {
            "log_path": str(log_path),
            "completed_epochs": 0,
            "dev_history": [],
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "log_path": str(log_path),
        "completed_epochs": len(MEAN_LOSS_PATTERN.findall(text)),
        "dev_history": [float(value) for value in DEV_HISTORY_PATTERN.findall(text)],
    }


def best_base_checkpoint(run_dir):
    candidates = []
    for path in run_dir.glob("dev_*_epoch*_model.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        candidates.append({
            "path": str(path),
            "dev_wer": float(match.group(1)),
            "epoch": int(match.group(2)),
        })
    return min(
        candidates,
        key=lambda value: (value["dev_wer"], value["epoch"], value["path"]),
        default=None,
    )


def calibrated_checkpoints(run_dir, prediction_dir=None):
    roots = [run_dir]
    if prediction_dir is not None and prediction_dir != run_dir:
        roots.append(prediction_dir)
    output = []
    for root in roots:
        for path in root.rglob("checkpoint_calibrated.pt"):
            item = {
                "path": str(path),
                "modified_ns": path.stat().st_mtime_ns,
            }
            if item not in output:
                output.append(item)
    return sorted(
        output,
        key=lambda value: value["modified_ns"],
        reverse=True,
    )


def find_prediction(run_dir, prediction_dir, mode):
    root = prediction_dir if prediction_dir is not None else run_dir
    matches = list(root.rglob("predictions-{}.jsonl".format(mode)))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def prediction_metrics(path, num_classes):
    if path is None:
        return None
    samples = read_predictions(path)
    sample_ids, edits, lengths = sentence_statistics(samples, num_classes)
    return {
        "path": str(path),
        "num_sentences": len(sample_ids),
        "edits": int(edits.sum()),
        "reference_occurrences": int(lengths.sum()),
        "wer": float(
            100.0 * edits.sum() / max(int(lengths.sum()), 1)
        ),
    }


def _read_calibration(path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as source:
        metrics = json.load(source)
    value = metrics.get("best_loss")
    if value is None:
        return None
    return {
        "path": str(path),
        "best_loss": float(value),
    }


def calibration_nll(run_dir, prediction_dir=None, calibration_path=None):
    if calibration_path is not None:
        result = _read_calibration(calibration_path)
        if result is None:
            raise ValueError(
                "calibration metrics do not exist or lack best_loss: {}".format(
                    calibration_path
                )
            )
        return result

    roots = [run_dir]
    if prediction_dir is not None and prediction_dir != run_dir:
        roots.insert(0, prediction_dir)
    for root in roots:
        candidates = [root / "fdgc_metrics.json"]
        if root.name == "final":
            candidates.append(root.parent / "fdgc" / "fdgc_metrics.json")
        elif root.name == "final_viterbi":
            candidates.append(root.parent / "fdgc" / "fdgc_metrics.json")
        elif root.name == "final_fdgc_fd":
            candidates.append(root.parent / "fdgc_fd" / "fdgc_metrics.json")
        candidates.extend([
            root.parent / "fdgc_metrics.json",
            root.parent / "fdgc" / "fdgc_metrics.json",
            root.parent / "fdgc_fd" / "fdgc_metrics.json",
            root.parent / "fdgc_ctc" / "fdgc_metrics.json",
        ])
        for path in candidates:
            result = _read_calibration(path)
            if result is not None:
                return result

    # A recursive search is safe only when there is exactly one metric file.
    # Multiple FDGC variants must be bound explicitly to avoid reporting the
    # calibration NLL of a different candidate.
    recursive = sorted(set(run_dir.rglob("fdgc_metrics.json")))
    if len(recursive) == 1:
        return _read_calibration(recursive[0])
    return None


def collect_results(run_specs, num_classes=1296):
    runs = {}
    for spec in run_specs:
        parsed = parse_run_spec(spec)
        name = parsed["name"]
        if name in runs:
            raise ValueError("duplicate run name: {}".format(name))
        run_dir = parsed["run_dir"]
        prediction_dir = parsed["prediction_dir"]
        calibration_path = parsed["calibration_path"]
        if not run_dir.is_dir():
            raise ValueError("run directory does not exist: {}".format(run_dir))
        training = parse_training_history(run_dir)
        checkpoint = best_base_checkpoint(run_dir)
        final_dev = prediction_metrics(
            find_prediction(run_dir, prediction_dir, "dev"),
            num_classes,
        )
        final_test = prediction_metrics(
            find_prediction(run_dir, prediction_dir, "test"),
            num_classes,
        )
        training_dev = min(training["dev_history"], default=None)
        selection_dev = (
            final_dev["wer"] if final_dev is not None else training_dev
        )
        runs[name] = {
            "run_dir": str(run_dir),
            "prediction_dir": (
                str(prediction_dir) if prediction_dir is not None else None
            ),
            "training": training,
            "best_base_checkpoint": checkpoint,
            "calibrated_checkpoints": calibrated_checkpoints(
                run_dir, prediction_dir
            ),
            "calibration": calibration_nll(
                run_dir,
                prediction_dir,
                calibration_path,
            ),
            "final_dev": final_dev,
            "final_test": final_test,
            "selection_dev_wer": selection_dev,
            "selection_eligible": bool(
                training["completed_epochs"] >= 40
                and final_dev is not None
            ),
        }

    eligible = {
        name: result["selection_dev_wer"]
        for name, result in runs.items()
        if result["selection_eligible"]
    }
    selected_by_dev = []
    dev_tied = []
    if eligible:
        best_dev = min(eligible.values())
        dev_tied = sorted(
            name for name, value in eligible.items() if value == best_dev
        )
        selected_by_dev = list(dev_tied)
        if len(dev_tied) > 1:
            calibration_values = {
                name: runs[name]["calibration"]["best_loss"]
                for name in dev_tied
                if runs[name]["calibration"] is not None
            }
            if len(calibration_values) == len(dev_tied):
                best_nll = min(calibration_values.values())
                selected_by_dev = sorted(
                    name
                    for name, value in calibration_values.items()
                    if value == best_nll
                )
    return {
        "selection_rule": (
            "rank by final D_sel Dev WER only after 40 completed epochs; "
            "break exact FDGC ties by D_cal calibration NLL; "
            "D_test is reported but never used for selection"
        ),
        "dev_tied": dev_tied,
        "selected_by_dev": selected_by_dev,
        "selected_dev_wer": (
            runs[selected_by_dev[0]]["selection_dev_wer"]
            if selected_by_dev
            else None
        ),
        "runs": runs,
    }


def markdown_report(result):
    lines = [
        "# Experiment Results",
        "",
        result["selection_rule"],
        "",
        "| Run | Epochs | Best checkpoint Dev | Final Dev | Final Test | Eligible |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name, run in result["runs"].items():
        checkpoint = run["best_base_checkpoint"]
        checkpoint_dev = (
            "{:.2f}%".format(checkpoint["dev_wer"])
            if checkpoint is not None
            else "-"
        )
        final_dev = run["final_dev"]
        final_test = run["final_test"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                name,
                run["training"]["completed_epochs"],
                checkpoint_dev,
                (
                    "{:.2f}%".format(final_dev["wer"])
                    if final_dev is not None
                    else "-"
                ),
                (
                    "{:.2f}%".format(final_test["wer"])
                    if final_test is not None
                    else "-"
                ),
                "yes" if run["selection_eligible"] else "no",
            )
        )
    lines.extend([
        "",
        "Selected by Dev (FDGC NLL tie-break): {}".format(
            ", ".join(result["selected_by_dev"])
            if result["selected_by_dev"]
            else "none"
        ),
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--num-classes", type=int, default=1296)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    result = collect_results(args.run, args.num_classes)
    json_path = Path(args.output_json)
    markdown_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
