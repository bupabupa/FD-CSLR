import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_candidate import (
    existing_test_predictions,
    load_optional_candidates,
    rank_candidates,
)


def _candidate(root, name, wer, nll=None, priority=0, test=False):
    run_dir = root / name
    prediction_dir = run_dir / "final"
    prediction_dir.mkdir(parents=True)
    prediction_path = prediction_dir / "predictions-dev.jsonl"
    prediction_path.write_text(
        json.dumps({
            "sample_id": "sample",
            "reference_ids": [1, 2, 3],
            "hypothesis_ids": [1],
        }) + "\n",
        encoding="utf-8",
    )
    (prediction_dir / "dev.txt").write_text(
        "Epoch 39, dev {:.2f}%\n".format(wer),
        encoding="utf-8",
    )
    config = root / "config.yaml"
    config.write_text("dataset: phoenix14\n", encoding="utf-8")
    checkpoint = run_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calibration = None
    if nll is not None:
        calibration = run_dir / "calibration.json"
        calibration.write_text(
            json.dumps({"best_loss": nll}),
            encoding="utf-8",
        )
    if test:
        (prediction_dir / "predictions-test.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
    return {
        "name": name,
        "dev_prediction": prediction_path,
        "run_dir": run_dir,
        "config": config,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "priority": priority,
    }


def test_dev_then_nll_then_priority_selection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidates = [
            _candidate(root, "higher_dev", 21.0, nll=0.1, priority=0),
            _candidate(root, "high_nll", 20.0, nll=0.3, priority=0),
            _candidate(root, "low_nll", 20.0, nll=0.2, priority=9),
            _candidate(root, "priority", 20.0, priority=1),
        ]
        result = rank_candidates(candidates, num_classes=4)
    assert result["selected"]["name"] == "low_nll"
    assert result["selected"]["dev_wer"] == 20.0
    assert result["selected"]["calibration_nll"] == 0.2


def test_exact_tie_without_nll_uses_declared_priority():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidates = [
            _candidate(root, "complex", 20.0, priority=5),
            _candidate(root, "simple", 20.0, priority=1),
        ]
        result = rank_candidates(candidates, num_classes=4)
    assert result["selected"]["name"] == "simple"


def test_prelock_test_prediction_is_detected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidate = _candidate(
            root, "leaked", 20.0, priority=1, test=True
        )
        existing = existing_test_predictions(candidate["run_dir"])
    assert existing
    assert existing[0].endswith("predictions-test.jsonl")


def test_optional_candidate_file_resolves_relative_paths():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidate = _candidate(
            root, "ngram", 20.0, nll=0.2, priority=75
        )
        optional_path = root / "optional_candidates.json"
        optional_path.write_text(
            json.dumps([{
                "name": candidate["name"],
                "dev": str(candidate["dev_prediction"].relative_to(root)),
                "run": str(candidate["run_dir"].relative_to(root)),
                "config": str(candidate["config"].relative_to(root)),
                "checkpoint": str(candidate["checkpoint"].relative_to(root)),
                "calibration": str(
                    candidate["calibration"].relative_to(root)
                ),
                "priority": candidate["priority"],
            }]),
            encoding="utf-8",
        )
        loaded = load_optional_candidates(optional_path)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "ngram"
    assert loaded[0]["priority"] == 75


if __name__ == "__main__":
    test_dev_then_nll_then_priority_selection()
    test_exact_tie_without_nll_uses_declared_priority()
    test_prelock_test_prediction_is_detected()
    test_optional_candidate_file_resolves_relative_paths()
    print("candidate lock tests passed")
