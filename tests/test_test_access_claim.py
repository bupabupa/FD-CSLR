import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import validate_test_access


def _write_claim(path):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(
            {
                "selected": {
                    "name": "candidate",
                    "dev_prediction": "/tmp/predictions-dev.jsonl",
                    "config": "/tmp/config.yaml",
                    "checkpoint": "/tmp/checkpoint.pt",
                }
            },
            output,
        )


def test_test_access_requires_claim_for_test_split():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        work_dir = root / "locked_test" / "candidate"
        args = SimpleNamespace(
            phase="test",
            load_splits=None,
            eval_splits=["test"],
            work_dir=str(work_dir),
        )
        try:
            validate_test_access(args)
        except RuntimeError as error:
            assert "test_access_claim.json" in str(error)
        else:
            raise AssertionError("Test access was allowed without a claim")

        claim_path = root / "test_access_claim.json"
        _write_claim(claim_path)
        assert validate_test_access(args) == claim_path


def test_calibration_views_do_not_require_test_claim():
    with tempfile.TemporaryDirectory() as temporary:
        args = SimpleNamespace(
            phase="test",
            load_splits=["cal", "dev"],
            eval_splits=["dev", "test"],
            work_dir=str(Path(temporary) / "calibration"),
        )
        assert validate_test_access(args) is None


def test_train_phase_does_not_require_test_claim():
    args = SimpleNamespace(
        phase="train",
        load_splits=None,
        eval_splits=["dev", "test"],
        work_dir="/tmp/train",
    )
    assert validate_test_access(args) is None


def test_malformed_claim_is_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        claim_path = root / "test_access_claim.json"
        claim_path.write_text("{}\n", encoding="utf-8")
        args = SimpleNamespace(
            phase="test",
            load_splits=["test"],
            eval_splits=None,
            work_dir=str(root / "run"),
        )
        try:
            validate_test_access(args)
        except RuntimeError as error:
            assert "missing selected candidate" in str(error)
        else:
            raise AssertionError("Malformed Test claim was accepted")


if __name__ == "__main__":
    test_test_access_requires_claim_for_test_split()
    test_calibration_views_do_not_require_test_claim()
    test_train_phase_does_not_require_test_claim()
    test_malformed_claim_is_rejected()
    print("test access claim tests passed")
