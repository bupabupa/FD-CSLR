import numpy as np
import torch

from evaluation.frequency_metrics import (
    aggregate_alignment,
    align_occurrences,
    class_frequency_report,
    frame_calibration_metrics,
    frequency_bins,
    frequency_metrics,
    gloss_occurrence_counts,
    occurrence_metrics,
    utterance_mean_wer,
)


def test_sequence_alignment():
    counts = align_occurrences(
        [1, 2, 3, 4, 1],
        [1, 4, 3, 4, 1],
        num_classes=5,
    )
    assert counts["tp"].tolist() == [0, 2, 0, 1, 1]
    assert counts["fn"].tolist() == [0, 0, 1, 0, 0]
    assert counts["fp"].tolist() == [0, 0, 0, 0, 1]
    aggregate = aggregate_alignment([[1, 2], [3]], [[1], [3, 4]], 5)
    assert aggregate["edits"] == 2
    assert aggregate["tp"].sum() == 2
    assert aggregate["fn"].sum() == 1
    assert aggregate["fp"].sum() == 1


def test_occurrence_and_frequency_metrics():
    counts = align_occurrences(
        [1, 2, 3, 4, 1],
        [1, 4, 3, 4, 1],
        num_classes=5,
    )
    metrics = occurrence_metrics(
        counts["tp"], counts["fp"], counts["fn"]
    )
    assert metrics["support"] == 5
    assert metrics["predicted"] == 5
    assert metrics["micro_f1"] > 0.7

    train_counts = [10, 6, 3, 1, 0]
    bins = frequency_bins(train_counts)
    assert bins["head"] == [0]
    assert bins["middle"] == [1]
    assert bins["tail"] == [2, 3, 4]
    by_frequency = frequency_metrics(
        counts["tp"],
        counts["fp"],
        counts["fn"],
        train_counts,
    )
    assert by_frequency["bins"]["tail"]["support"] == 3


def test_semantic_class_report_and_training_counts():
    gloss_dict = {"a": [1], "b": [2], "c": [3]}
    training_counts = gloss_occurrence_counts(
        [
            {"label": "a a b"},
            {"label": "a c"},
        ],
        gloss_dict,
        num_classes=4,
    )
    assert training_counts.tolist() == [0, 3, 1, 1]

    report = class_frequency_report(
        [[1, 2], [3]],
        [[1, 2], [2]],
        training_counts,
        blank_id=0,
    )
    assert report["overall"]["support"] == 3
    assert report["overall"]["true_positive"] == 2
    assert report["per_class"]["class_ids"] == [1, 2, 3]
    assert report["per_class"]["error_rate"][0] == 0
    assert report["per_class"]["uwer_error_rate"][0] == 0
    assert len(report["per_class"]["precision"]) == 3
    assert len(report["per_class"]["recall"]) == 3
    assert len(report["per_class"]["f1"]) == 3
    assert report["overall"]["uwer"] > 0.0
    assert abs(
        sum(report["per_class"]["uwer_error_rate"]) / 3.0
        - report["overall"]["uwer"]
    ) < 1e-12
    assert report["utterance_mean_wer"] == 0.5
    assert all(
        values["num_classes"] == 1
        for values in report["bins"].values()
    )


def test_frame_calibration_metrics():
    log_probs = torch.tensor(
        [
            [0.0, -10.0, -10.0],
            [-10.0, 0.0, -10.0],
            [-10.0, -10.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 1, 2])
    metrics = frame_calibration_metrics(
        log_probs,
        targets,
        source_prior=torch.tensor([0.6, 0.3, 0.1]),
        target_prior=torch.ones(3) / 3,
    )
    assert metrics["nll"] == 0
    assert metrics["ece"] == 0
    assert metrics["accuracy"] == 1
    assert abs(metrics["brier"]) < 1e-6
    assert metrics["prior_weighted_ece"] == 0


def test_per_class_error_rate_is_bounded_confusion_rate():
    report = class_frequency_report(
        references=[[1, 2, 2]],
        hypotheses=[[1, 2, 3]],
        train_counts=[0, 10, 5, 1],
        blank_id=0,
    )
    torch.testing.assert_close(
        torch.tensor(report["per_class"]["error_rate"]),
        torch.tensor([0.0, 0.5, 1.0]),
    )
    torch.testing.assert_close(
        torch.tensor(report["per_class"]["uwer_error_rate"]),
        torch.tensor([0.0, 0.5, 0.0]),
    )
    assert report["overall"]["uwer"] == 0.25
    assert utterance_mean_wer([[1, 2], [3]], [[1, 2], [2]], 4) == 0.5


if __name__ == "__main__":
    test_sequence_alignment()
    test_occurrence_and_frequency_metrics()
    test_semantic_class_report_and_training_counts()
    test_frame_calibration_metrics()
    test_per_class_error_rate_is_bounded_confusion_rate()
    print("frequency metric tests passed")
