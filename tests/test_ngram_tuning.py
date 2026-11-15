import numpy as np

from scripts.tune_ngram_rescorer import (
    choose_beam,
    evaluate_selection,
    parse_float_list,
    tune_grid,
)
from utils.ngram_rescorer import NGramRescorer


def test_grid_parser_rejects_empty_values():
    try:
        parse_float_list(" , ")
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty grid was accepted")


def test_ngram_rescoring_can_reorder_mocked_nbest():
    scorer = NGramRescorer(
        [[1, 2], [1, 2], [1, 2], [2, 1]],
        order=2,
        vocabulary_size=4,
    )
    beams = [
        {"tokens": [2, 1], "ctc_score": 0.01},
        {"tokens": [1, 2], "ctc_score": 0.10},
    ]
    assert choose_beam(beams, scorer, alpha=0.0, beta=0.0) == 0
    assert choose_beam(beams, scorer, alpha=1.0, beta=0.0) == 1


def test_wer_uses_edit_distance_with_length_changes():
    scorer = NGramRescorer(
        [[1, 2]],
        order=2,
        vocabulary_size=4,
    )
    records = [{
        "reference_ids": [1, 2],
        "beams": [{"tokens": [1], "ctc_score": 0.0}],
    }]
    result = evaluate_selection(
        records,
        scorer,
        alpha=0.0,
        beta=0.0,
        train_counts=np.zeros(4, dtype=np.int64),
        num_classes=4,
    )
    assert result["edits"] == 1
    assert result["reference_occurrences"] == 2
    assert result["wer"] == 50.0


def test_external_wer_evaluator_overrides_authoritative_score():
    scorer = NGramRescorer(
        [[1, 2]],
        order=2,
        vocabulary_size=4,
    )
    records = [{
        "reference_ids": [1, 2],
        "beams": [
            {"tokens": [1], "ctc_score": 0.0},
            {"tokens": [1, 2], "ctc_score": 1.0},
        ],
    }]
    result = tune_grid(
        records,
        scorer,
        alphas=[0.0],
        betas=[0.0],
        train_counts=np.zeros(4, dtype=np.int64),
        num_classes=4,
        wer_evaluator=lambda hypotheses: 12.5,
    )
    assert result["baseline"]["wer"] == 12.5
    assert result["baseline"]["python_edit_wer"] == 50.0
    assert result["baseline"]["wer_source"] == "sclite"


if __name__ == "__main__":
    test_grid_parser_rejects_empty_values()
    test_ngram_rescoring_can_reorder_mocked_nbest()
    test_wer_uses_edit_distance_with_length_changes()
    test_external_wer_evaluator_overrides_authoritative_score()
    print("n-gram tuning tests passed")
