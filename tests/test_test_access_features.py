import tempfile
from pathlib import Path
from types import SimpleNamespace

from main import validate_test_access


def test_features_phase_requires_test_claim():
    with tempfile.TemporaryDirectory() as temporary:
        arg = SimpleNamespace(
            phase="features",
            load_splits=None,
            eval_splits=(),
            work_dir=temporary,
        )
        try:
            validate_test_access(arg)
        except RuntimeError as error:
            assert "test_access_claim.json" in str(error)
        else:
            raise AssertionError("features phase bypassed the Test gate")


def test_train_phase_does_not_require_test_claim():
    with tempfile.TemporaryDirectory() as temporary:
        arg = SimpleNamespace(
            phase="train",
            load_splits=None,
            eval_splits=(),
            work_dir=temporary,
        )
        assert validate_test_access(arg) is None


if __name__ == "__main__":
    test_features_phase_requires_test_claim()
    test_train_phase_does_not_require_test_claim()
    print("Test access feature-gate tests passed")
