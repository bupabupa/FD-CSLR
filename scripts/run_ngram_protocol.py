import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "python3"
DEV_WER_PATTERN = re.compile(
    r"\bdev\s+([0-9]+(?:\.[0-9]+)?)%",
    flags=re.IGNORECASE,
)


def pid_is_alive(pid):
    if pid is None:
        return True
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def wait_for_completion(status_path, marker, pid, interval=60):
    status_path = Path(status_path)
    while True:
        if status_path.is_file():
            text = status_path.read_text(encoding="utf-8", errors="replace")
            if marker in text:
                return
        if not pid_is_alive(pid):
            raise RuntimeError(
                "split protocol exited before marker {}".format(marker)
            )
        time.sleep(interval)


def best_checkpoint(run_dir):
    candidates = []
    for path in Path(run_dir).glob("dev_*_epoch*_model.pt"):
        match = re.match(
            r"^dev_([0-9]+(?:\.[0-9]+)?)_epoch([0-9]+)_model\.pt$",
            path.name,
        )
        if match:
            candidates.append((
                float(match.group(1)),
                int(match.group(2)),
                path,
            ))
    if not candidates:
        raise FileNotFoundError(
            "no Dev checkpoint found in {}".format(run_dir)
        )
    return min(candidates, key=lambda item: (item[0], item[1], str(item[2])))[2]


def make_rescored_config(base_config, output, rescorer_path, alpha, beta):
    with open(base_config, encoding="utf-8") as source:
        payload = yaml.load(source, Loader=yaml.FullLoader)
    model_args = dict(payload.get("model_args", {}))
    model_args.update({
        "beam_width": 50,
        "num_processes": 4,
        "rescorer_path": str(Path(rescorer_path).resolve()),
        "rescorer_alpha": float(alpha),
        "rescorer_beta": float(beta),
    })
    payload["model_args"] = model_args
    payload["split_manifest"] = str(
        (ROOT / payload["split_manifest"]).resolve()
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as destination:
        yaml.safe_dump(payload, destination, sort_keys=False)
    return output


def register_optional_candidate(path, candidate):
    path = Path(path)
    payload = []
    if path.is_file():
        with open(path, encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, list):
            raise ValueError("optional candidate file must contain a list")
    payload = [
        item for item in payload
        if item.get("name") != candidate["name"]
    ]
    payload.append(candidate)
    payload.sort(key=lambda item: item["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
    temporary.replace(path)
    return path


def last_dev_wer(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = DEV_WER_PATTERN.findall(text)
    if not matches:
        raise RuntimeError("Dev WER was not written to {}".format(path))
    return float(matches[-1])


def run_logged(command, log_path):
    with open(log_path, "a", encoding="utf-8") as output:
        output.write("$ {}\n".format(" ".join(map(str, command))))
        output.flush()
        subprocess.run(
            [str(value) for value in command],
            check=True,
            cwd=str(ROOT),
            stdout=output,
            stderr=subprocess.STDOUT,
        )


def write_status(path, message):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    line = "[{}] {}\n".format(timestamp, message)
    with open(path, "a", encoding="utf-8") as destination:
        destination.write(line)
    print(line, end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-status", required=True)
    parser.add_argument("--split-pid", type=int)
    parser.add_argument(
        "--dependency-marker",
        default="PROTOCOL_COMPLETE",
        help="marker required in the dependency status file",
    )
    parser.add_argument(
        "--status-dir",
        default=str(
            ROOT / "work_dir" / "fd_xmp_v3" / "ngram_protocol"
        ),
    )
    args = parser.parse_args()
    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "status.txt"
    log_path = status_dir / "protocol.log"

    try:
        write_status(
            status_path,
            "WAIT dependency marker={} status={}".format(
                args.dependency_marker,
                args.split_status,
            ),
        )
        wait_for_completion(
            args.split_status,
            args.dependency_marker,
            args.split_pid,
        )
        base_run = ROOT / "work_dir/fd_xmp_v3/E1_fdxmp_v3_split_seed0"
        calibrated_checkpoint = base_run / "fdgc/checkpoint_calibrated.pt"
        checkpoint = (
            calibrated_checkpoint
            if calibrated_checkpoint.is_file()
            else best_checkpoint(base_run)
        )
        base_config = ROOT / "configs/fdxmp_v3_split.yaml"
        manifest = ROOT / "preprocess/phoenix2014/split_manifest.json"
        gloss_dict = ROOT / "preprocess/phoenix2014/gloss_dict.npy"
        rescorer_path = (
            ROOT
            / "work_dir/fd_xmp_v3/ngram/strict_train_order3.json"
        )
        ngram_dir = base_run / "ngram"
        ngram_dir.mkdir(parents=True, exist_ok=True)
        nbest_path = ngram_dir / "selection_dev_nbest50.json"
        tuning_path = ngram_dir / "tuning_sclite.json"
        offline_predictions = ngram_dir / "offline_predictions-dev.jsonl"

        write_status(status_path, "START strict Dev n-best")
        run_logged([
            PYTHON,
            "scripts/cache_ctc_nbest.py",
            "--config",
            base_config,
            "--load-weights",
            checkpoint,
            "--output",
            nbest_path,
            "--run-dir",
            ngram_dir / "cache_run",
            "--device",
            "0",
            "--batch-size",
            "4",
            "--num-workers",
            "2",
            "--beam-width",
            "50",
            "--num-processes",
            "4",
        ], log_path)
        write_status(status_path, "COMPLETE strict Dev n-best")

        write_status(status_path, "START SCLITE grid")
        run_logged([
            PYTHON,
            "scripts/tune_ngram_rescorer.py",
            "--nbest",
            nbest_path,
            "--rescorer",
            rescorer_path,
            "--gloss-dict",
            gloss_dict,
            "--split-manifest",
            manifest,
            "--alphas",
            "0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1",
            "--betas",
            "0",
            "--output",
            tuning_path,
            "--prediction-output",
            offline_predictions,
        ], log_path)
        with open(tuning_path, encoding="utf-8") as source:
            tuning = json.load(source)
        selected = tuning.get("selected")
        if selected is None:
            write_status(
                status_path,
                "NGRAM_PROTOCOL_COMPLETE no_eligible_candidate",
            )
            return
        write_status(
            status_path,
            "COMPLETE SCLITE grid alpha={} beta={} Dev={}".format(
                selected["alpha"],
                selected["beta"],
                selected["wer"],
            ),
        )

        candidate_config = make_rescored_config(
            base_config,
            ngram_dir / "config.yaml",
            rescorer_path,
            selected["alpha"],
            selected["beta"],
        )
        final_dir = ngram_dir / "final"
        write_status(status_path, "START model-integrated Dev evaluation")
        run_logged([
            PYTHON,
            "-u",
            "main.py",
            "--device",
            "0",
            "--phase",
            "test",
            "--work-dir",
            final_dir,
            "--config",
            candidate_config,
            "--load-weights",
            checkpoint,
            "--eval-splits",
            "dev",
            "--evaluate-tool",
            "sclite",
        ], log_path)
        integrated_wer = last_dev_wer(final_dir / "dev.txt")
        if abs(integrated_wer - float(selected["wer"])) > 1e-9:
            raise RuntimeError(
                "offline and integrated Dev WER differ: {} vs {}".format(
                    selected["wer"], integrated_wer
                )
            )
        calibration_path = base_run / "fdgc/fdgc_metrics.json"
        candidate = {
            "name": "fdxmp_ngram_calibrated",
            "dev": str((final_dir / "predictions-dev.jsonl").resolve()),
            "run": str(base_run.resolve()),
            "config": str(candidate_config.resolve()),
            "checkpoint": str(Path(checkpoint).resolve()),
            "calibration": (
                str(calibration_path.resolve())
                if calibration_path.is_file() else None
            ),
            "priority": 80,
        }
        optional_path = (
            ROOT
            / "work_dir/fd_xmp_v3/candidate_protocol"
            / "optional_candidates.json"
        )
        register_optional_candidate(optional_path, candidate)
        write_status(
            status_path,
            "NGRAM_PROTOCOL_COMPLETE Dev={} optional={}".format(
                integrated_wer,
                optional_path,
            ),
        )
    except Exception as error:
        write_status(status_path, "FAILED {}".format(error))
        raise


if __name__ == "__main__":
    main()
