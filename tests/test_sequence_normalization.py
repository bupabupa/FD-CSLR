from pathlib import Path

from sequence_normalization import (
    aggregate_wer,
    normalize_token_sequences,
)


PREPROCESS = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "slr_eval"
    / "preprocess.sh"
)


def test_normalization_matches_vac_token_cleanup():
    normalized = normalize_token_sequences(
        [["loc-NORD", "__PU__", "S0NNE"]],
        PREPROCESS,
    )
    assert normalized == [["NORD", "SONNE"]]


def test_aggregate_wer_uses_sentence_edits():
    result = aggregate_wer(
        [["A", "B"], ["C"]],
        [["A", "D"], ["C"]],
    )
    assert result["edits"] == 1
    assert result["reference_length"] == 3
    assert abs(result["wer"] - 100.0 / 3.0) < 1e-9


if __name__ == "__main__":
    test_normalization_matches_vac_token_cleanup()
    test_aggregate_wer_uses_sentence_edits()
    print("sequence normalization tests passed")
