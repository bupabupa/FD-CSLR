from pathlib import Path

import numpy as np

from evaluation.frequency_metrics import (
    class_frequency_report,
    frequency_bins,
    gloss_occurrence_counts,
)


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "preprocess" / "phoenix2014"


def test_phoenix_training_frequency_counts():
    gloss_dict = np.load(
        PREPROCESS / "gloss_dict.npy", allow_pickle=True
    ).item()
    train_info = np.load(
        PREPROCESS / "train_info.npy", allow_pickle=True
    ).item()
    counts = gloss_occurrence_counts(train_info, gloss_dict)

    assert len(counts) == len(gloss_dict) + 1
    assert counts[0] == 0
    assert counts.sum() > len(train_info)
    bins = frequency_bins(counts[1:])
    assert all(bin_ids for bin_ids in bins.values())

    report = class_frequency_report(
        [[1]],
        [[1]],
        counts,
        blank_id=0,
    )
    assert report["overall"]["true_positive"] == 1


if __name__ == "__main__":
    test_phoenix_training_frequency_counts()
    print("frequency dataset test passed")
