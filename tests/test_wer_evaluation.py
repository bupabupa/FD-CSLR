import tempfile
from pathlib import Path

from evaluation.slr_eval.wer_calculation import (
    _parse_sclite_wer,
    _sort_stm_by_document,
)


def test_sort_preserves_line_boundaries():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        source = directory / "source.stm"
        destination = directory / "sorted.stm"
        source.write_text(
            "b_doc 1 signer01 0.0 1.0 B\n"
            "a_doc 1 signer02 0.0 1.0 A",
            encoding="utf-8",
        )
        _sort_stm_by_document(source, destination)
        assert destination.read_text(encoding="utf-8").splitlines() == [
            "a_doc 1 signer02 0.0 1.0 A",
            "b_doc 1 signer01 0.0 1.0 B",
        ]


def test_parse_sclite_error():
    output = (
        "Aligning transcripts\n"
        "Percent Total Error = 41.2%\n"
    )
    assert _parse_sclite_wer(output) == 41.2


if __name__ == "__main__":
    test_sort_preserves_line_boundaries()
    test_parse_sclite_error()
    print("WER evaluation tests passed")
