import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ngram_protocol import (
    PYTHON,
    best_checkpoint,
    last_dev_wer,
    make_rescored_config,
    register_optional_candidate,
    run_logged,
    write_status,
)


MANIFEST = ROOT / "preprocess/phoenix2014/split_manifest.json"
GLOSS_DICT = ROOT / "preprocess/phoenix2014/gloss_dict.npy"
RESCORER = ROOT / "work_dir/fd_xmp_v3/ngram/strict_train_order3.json"
OPTIONAL_PATH = (
    ROOT
    / "work_dir/fd_xmp_v3/candidate_protocol/optional_candidates.json"
)


def resolve_checkpoint(base_run, checkpoint):
    if checkpoint:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError("checkpoint does not exist: {}".format(path))
        return path
    calibrated = Path(base_run) / "fdgc/checkpoint_calibrated.pt"
    if calibrated.is_file():
        return calibrated.resolve()
    return best_checkpoint(base_run).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--status-dir", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-priority", type=int, default=80)
    args = parser.parse_args()

    base_run = Path(args.base_run).resolve()
    base_config = Path(args.base_config).resolve()
    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "status.txt"
    log_path = status_dir / "protocol.log"

    try:
        checkpoint = resolve_checkpoint(base_run, args.checkpoint)
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
            RESCORER,
            "--gloss-dict",
            GLOSS_DICT,
            "--split-manifest",
            MANIFEST,
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
            RESCORER,
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
            "name": args.candidate_name,
            "dev": str((final_dir / "predictions-dev.jsonl").resolve()),
            "run": str(base_run),
            "config": str(candidate_config.resolve()),
            "checkpoint": str(checkpoint),
            "calibration": (
                str(calibration_path.resolve())
                if calibration_path.is_file() else None
            ),
            "priority": int(args.candidate_priority),
        }
        register_optional_candidate(OPTIONAL_PATH, candidate)
        write_status(
            status_path,
            "NGRAM_PROTOCOL_COMPLETE Dev={} candidate={}".format(
                integrated_wer,
                args.candidate_name,
            ),
        )
    except Exception as error:
        write_status(status_path, "FAILED {}".format(error))
        raise


if __name__ == "__main__":
    main()
