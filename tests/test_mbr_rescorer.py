import numpy as np

from scripts.tune_mbr_rescorer import (
    edit_distance,
    evaluate_ctc_baseline,
    evaluate_selection,
    mbr_select,
    normalized_distances,
    tune_grid,
)


def test_edit_distance_and_normalized_distance():
    assert edit_distance([1, 2, 3], [1, 3]) == 1
    distances = normalized_distances([[1, 2], [1, 2, 3]], "max")
    assert distances[0, 1] == 1.0 / 3.0
    assert distances[1, 0] == 1.0 / 3.0


def test_mbr_prefers_consensus_candidate():
    beams = [
        {"tokens": [1, 2], "ctc_score": 0.0},
        {"tokens": [1, 2, 3], "ctc_score": 0.1},
        {"tokens": [1, 3], "ctc_score": 0.2},
    ]
    assert mbr_select(
        beams,
        temperature=10.0,
        ctc_weight=0.0,
    ) == 1


def test_ctc_weight_preserves_first_beam_without_consensus():
    beams = [
        {"tokens": [1], "ctc_score": 0.0},
        {"tokens": [1, 2], "ctc_score": 1.0},
    ]
    assert mbr_select(
        beams,
        temperature=1.0,
        ctc_weight=100.0,
    ) == 0


def test_ctc_baseline_uses_minimum_ctc_score():
    records = [{
        "reference_ids": [1],
        "beams": [
            {"tokens": [2], "ctc_score": 0.2},
            {"tokens": [1], "ctc_score": 0.1},
        ],
    }]
    result = evaluate_ctc_baseline(
        records,
        train_counts=np.zeros(4, dtype=np.int64),
        num_classes=4,
    )
    assert result["selected_ctc_beam_counts"] == {"1": 1}
    assert result["hypotheses"] == [[1]]


def test_wer_guard_rejects_harmful_mbr_selection():
    records = [{
        "reference_ids": [1, 2],
        "beams": [
            {"tokens": [1, 2], "ctc_score": 0.0},
            {"tokens": [1], "ctc_score": 0.1},
        ],
    }]
    result = tune_grid(
        records,
        temperatures=[1.0],
        ctc_weights=[0.0],
        denominators=["max"],
        train_counts=np.zeros(4, dtype=np.int64),
        num_classes=4,
        wer_evaluator=lambda hypotheses: 0.0,
    )
    assert result["selected"] is None


if __name__ == "__main__":
    test_edit_distance_and_normalized_distance()
    test_mbr_prefers_consensus_candidate()
    test_ctc_weight_preserves_first_beam_without_consensus()
    test_ctc_baseline_uses_minimum_ctc_score()
    test_wer_guard_rejects_harmful_mbr_selection()
    print("MBR rescoring tests passed")
