import json
import tempfile
from pathlib import Path

import torch

from modules.calibration import FDGCCalibrator
from scripts.train_fdgc import (
    _ece,
    _edit_distance,
    configure_difficulty,
    ctc_calibration_loss,
    difficulty_from_error_counts,
    estimate_calibration_difficulty,
    evaluate_dev_wer,
    extract_alignment_targets,
    is_better_calibration,
    is_better_fdgc,
    load_difficulty,
)


def _collapse(path):
    output = []
    previous = None
    for class_id in path:
        if class_id != 0 and class_id != previous:
            output.append(class_id)
        previous = class_id
    return output


class _DummyDataset:
    dict = {"a": [1], "b": [2]}


class _DummyLoader:
    dataset = _DummyDataset()

    def __iter__(self):
        yield (
            torch.zeros(1, 3),
            torch.tensor([3]),
            torch.tensor([1, 2]),
            torch.tensor([2]),
            torch.tensor([4]),
            ("sample|sample|Signer04|sample",),
        )

    def __len__(self):
        return 1


class _DummyDevice:
    @staticmethod
    def data_to_device(value):
        return value


class _DummyModel:
    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        return {"recognized_sents": [[("a", 0), ("unknown", 1)]]}


class _DifficultyLoader:
    def __iter__(self):
        yield (
            torch.zeros(1, 3),
            torch.tensor([5]),
            torch.tensor([1, 2]),
            torch.tensor([2]),
            torch.tensor([4]),
            ("sample|sample|Signer04|sample",),
        )


class _DifficultyModel:
    supports_signer = True

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        logits = torch.full((5, 1, 4), -5.0)
        logits[:, 0, 1] = 5.0
        return {"sequence_logits": logits, "feat_len": torch.tensor([5])}


def test_alignment_targets_and_difficulty():
    torch.manual_seed(71)
    logits = torch.randn(5, 2, 4)
    labels = torch.tensor([1, 2, 1])
    label_lengths = torch.tensor([2, 1])
    feature_lengths = torch.tensor([5, 3])
    targets, mask = extract_alignment_targets(
        logits, labels, label_lengths, feature_lengths
    )
    assert mask[:, 0].tolist() == [True] * 5
    assert mask[:, 1].tolist() == [True, True, True, False, False]
    assert _collapse(targets[:, 0].tolist()) == [1, 2]
    assert _collapse(targets[:, 1].tolist()) == [1]

    with tempfile.TemporaryDirectory() as directory:
        report_path = Path(directory) / "frequency.json"
        with open(report_path, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "per_class": {
                        "class_ids": [1, 2, 3],
                        "error_rate": [0.1, 0.2, 0.0],
                    }
                },
                output,
            )
        difficulty = load_difficulty(report_path, 3)
        no_difficulty = configure_difficulty(
            report_path, 3, use_difficulty=False
        )
        with_difficulty = configure_difficulty(
            report_path, 3, use_difficulty=True
        )
    torch.testing.assert_close(
        difficulty, torch.tensor([0.5, 1.0, 0.0])
    )

    confidence = torch.tensor([1.0, 0.0])
    correct = torch.tensor([True, False])
    assert float(_ece(
        confidence, correct, torch.ones_like(confidence)
    )) < 1e-6
    assert _edit_distance([1, 2, 3], [1, 4, 3]) == 1
    assert _edit_distance([1, 2], [1, 2, 3]) == 1
    assert _edit_distance([1, 2], [1]) == 1
    assert evaluate_dev_wer(
        _DummyModel(), _DummyLoader(), _DummyDevice(), dataset_info=None
    ) == 50.0
    assert is_better_fdgc(39.0, 2.0, 40.0, 1.0)
    assert not is_better_fdgc(40.0, 1.1, 40.0, 1.0)
    assert is_better_fdgc(40.0, 0.9, 40.0, 1.0)
    assert not is_better_fdgc(41.0, 0.1, 40.0, 1.0)
    assert is_better_calibration(1.0, 0.2, 1.1, 0.1)
    assert not is_better_calibration(1.1, 0.0, 1.0, 0.1)
    assert is_better_calibration(1.0, 0.09, 1.0, 0.1)
    assert torch.equal(
        no_difficulty,
        torch.zeros(3),
    )
    assert torch.equal(
        with_difficulty,
        torch.tensor([0.5, 1.0, 0.0]),
    )


def test_calibration_difficulty_is_computed_from_error_counts():
    difficulty, statistics = difficulty_from_error_counts(
        torch.tensor([1.0, 0.0, 0.0, 2.0]),
        torch.tensor([2.0, 4.0, 0.0, 2.0]),
    )
    torch.testing.assert_close(
        difficulty,
        torch.tensor([0.5, 0.0, 0.0, 1.0], dtype=torch.float64),
    )
    assert statistics["source"] == "D_cal_viterbi_error"
    assert statistics["num_frames"] == 8
    assert statistics["num_covered_classes"] == 3
    assert statistics["num_missing_classes"] == 1

    calibration_difficulty, calibration_statistics = (
        estimate_calibration_difficulty(
            _DifficultyModel(),
            _DifficultyLoader(),
            _DummyDevice(),
            3,
        )
    )
    assert calibration_difficulty.shape == (3,)
    assert torch.isfinite(calibration_difficulty).all()
    assert calibration_statistics["source"] == "D_cal_viterbi_error"
    assert calibration_statistics["num_frames"] > 0


def test_impossible_alignment_is_excluded_from_fdgc_targets():
    logits = torch.randn(3, 2, 4)
    labels = torch.tensor([1, 2, 2, 2])
    label_lengths = torch.tensor([1, 3])
    feature_lengths = torch.tensor([3, 3])

    targets, mask = extract_alignment_targets(
        logits,
        labels,
        label_lengths,
        feature_lengths,
    )

    assert mask[:, 0].all()
    assert not mask[:, 1].any()
    assert torch.count_nonzero(targets[:, 1]) == 0


def test_ctc_calibration_loss_is_sequence_level():
    logits = torch.full((7, 1, 4), -5.0)
    with torch.no_grad():
        logits[:, 0, 0] = 5.0
    calibrator = FDGCCalibrator(num_classes=4)
    output = calibrator(
        logits,
        source_prior=torch.tensor([0.5, 0.3, 0.2]),
    )
    labels = torch.tensor([1, 2])
    label_lengths = torch.tensor([2])
    feature_lengths = torch.tensor([7])

    loss = ctc_calibration_loss(
        output,
        labels,
        label_lengths,
        feature_lengths,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert calibrator.class_mlp[-1].weight.grad is not None
    assert calibrator.frame_mlp[-1].weight.grad is not None
    assert calibrator.class_mlp[-1].weight.grad.abs().sum().item() > 0
    assert calibrator.frame_mlp[-1].weight.grad.abs().sum().item() > 0


def test_ctc_calibration_regularizes_only_valid_frames():
    log_probs = torch.full((4, 1, 3), -torch.log(torch.tensor(3.0)))
    tau = torch.tensor(
        [
            [[1.0, 1.0, 1.0]],
            [[2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0]],
            [[100.0, 100.0, 100.0]],
        ]
    )
    output = {"log_probs": log_probs, "tau": tau}
    labels = torch.tensor([1, 2])
    label_lengths = torch.tensor([2])
    feature_lengths = torch.tensor([3])

    base_loss = ctc_calibration_loss(
        output,
        labels,
        label_lengths,
        feature_lengths,
        regularization=0.0,
    )
    regularized_loss = ctc_calibration_loss(
        output,
        labels,
        label_lengths,
        feature_lengths,
        regularization=0.01,
    )
    expected_penalty = 0.01 * tau[:3].square().mean()
    torch.testing.assert_close(
        regularized_loss - base_loss,
        expected_penalty,
    )


def test_ctc_calibration_skips_an_impossible_sample():
    log_probs = torch.full((7, 2, 4), -20.0)
    log_probs[:, 0, 0] = 0.0
    log_probs[:, 1, 0] = 0.0
    output = {
        "log_probs": log_probs,
        "tau": torch.zeros(7, 2, 3),
    }
    labels = torch.tensor([1, 2, 2, 2, 2])
    label_lengths = torch.tensor([2, 3])
    feature_lengths = torch.tensor([7, 2])

    loss = ctc_calibration_loss(
        output,
        labels,
        label_lengths,
        feature_lengths,
    )

    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_alignment_targets_and_difficulty()
    test_calibration_difficulty_is_computed_from_error_counts()
    test_impossible_alignment_is_excluded_from_fdgc_targets()
    test_ctc_calibration_loss_is_sequence_level()
    test_ctc_calibration_regularizes_only_valid_frames()
    test_ctc_calibration_skips_an_impossible_sample()
    print("FDGC training tests passed")
