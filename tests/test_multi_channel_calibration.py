import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.calibration import (
    MultiChannelCalibrator,
    validate_fusion_state,
)


def test_identical_channels_preserve_normalized_probabilities():
    logits = torch.randn(6, 2, 5)
    calibrator = MultiChannelCalibrator(num_classes=5)
    output = calibrator(logits, logits.clone(), logits.clone())
    torch.testing.assert_close(
        output["probabilities"],
        logits.softmax(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        output["probabilities"].sum(dim=-1),
        torch.ones(6, 2),
    )
    torch.testing.assert_close(
        calibrator.weights().sum(),
        torch.tensor(1.0),
    )
    assert calibrator.weights()[0].item() > 0.99
    assert calibrator.fusion_parameterization_version.item() == 2


def test_initial_fusion_is_anchored_to_the_linear_channel():
    torch.manual_seed(29)
    linear = torch.randn(6, 2, 5)
    visual = torch.randn(6, 2, 5)
    text = torch.randn(6, 2, 5)
    calibrator = MultiChannelCalibrator(num_classes=5)
    output = calibrator(linear, visual, text)
    torch.testing.assert_close(
        output["probabilities"],
        linear.softmax(dim=-1),
        atol=6e-3,
        rtol=0.0,
    )


def test_three_channel_calibration_loss_and_gradients():
    linear = torch.randn(8, 2, 6)
    visual = torch.randn(8, 2, 6)
    text = torch.randn(8, 2, 6)
    calibrator = MultiChannelCalibrator(num_classes=6)
    output = calibrator(linear, visual, text)
    targets = torch.zeros(8, 2, dtype=torch.long)
    frame_mask = torch.ones(8, 2, dtype=torch.bool)
    frame_mask[-1, -1] = False
    loss = calibrator.calibration_loss(output, targets, frame_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert calibrator.log_temperatures.grad is not None
    assert calibrator.fusion_logits.grad is not None
    assert calibrator.log_temperatures.grad.abs().sum().item() > 0
    assert calibrator.fusion_logits.grad.abs().sum().item() > 0


def test_rejects_nonfinite_channel():
    calibrator = MultiChannelCalibrator(num_classes=4)
    linear = torch.zeros(3, 1, 4)
    visual = torch.zeros(3, 1, 4)
    text = torch.zeros(3, 1, 4)
    text[0, 0, 0] = float("nan")
    try:
        calibrator(linear, visual, text)
    except FloatingPointError:
        return
    raise AssertionError("non-finite channel was not rejected")


def test_fusion_state_version_is_required_and_exact():
    calibrator = MultiChannelCalibrator(num_classes=4)
    state = calibrator.state_dict()
    validate_fusion_state(state)
    for invalid in (
        {},
        {"fusion_parameterization_version": torch.tensor(1)},
        {"fusion_parameterization_version": torch.tensor(3)},
    ):
        try:
            validate_fusion_state(invalid)
        except ValueError:
            continue
        raise AssertionError("invalid fusion state version was accepted")


if __name__ == "__main__":
    test_identical_channels_preserve_normalized_probabilities()
    test_initial_fusion_is_anchored_to_the_linear_channel()
    test_three_channel_calibration_loss_and_gradients()
    test_rejects_nonfinite_channel()
    test_fusion_state_version_is_required_and_exact()
    print("multi-channel calibration tests passed")
