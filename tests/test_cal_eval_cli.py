import tempfile
from pathlib import Path

import utils
from main import validate_test_access
from seq_scripts import _sequence_wer


PREPROCESS = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "slr_eval"
    / "preprocess.sh"
)


def test_cal_is_a_valid_main_eval_split():
    args = utils.get_parser().parse_args([
        "--phase",
        "test",
        "--eval-splits",
        "cal",
    ])
    assert args.eval_splits == ["cal"]
    with tempfile.TemporaryDirectory() as temporary:
        args.work_dir = temporary
        args.load_splits = None
        assert validate_test_access(args) is None


def test_cal_wer_uses_manifest_aligned_sequences():
    wer = _sequence_wer(
        [["A", "B"], ["C"]],
        [["A", "D"], ["C"]],
        PREPROCESS,
    )
    assert abs(wer - 100.0 / 3.0) < 1e-9


if __name__ == "__main__":
    test_cal_is_a_valid_main_eval_split()
    test_cal_wer_uses_manifest_aligned_sequences()
    print("cal evaluation CLI tests passed")
