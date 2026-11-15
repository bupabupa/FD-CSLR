import torch
import torch.nn.functional as F

from modules.calibration import FDGCCalibrator, _top_margin


def test_identity_initialization_and_normalization():
    torch.manual_seed(11)
    calibrator = FDGCCalibrator(num_classes=6, blank_id=0)
    logits = torch.randn(7, 2, 6)
    source_prior = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])
    target_prior = torch.full_like(source_prior, 0.2)

    output = calibrator(logits, source_prior, target_prior)

    torch.testing.assert_close(output["logits"], logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        output["log_probs"].exp().sum(dim=-1),
        torch.ones(7, 2),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.count_nonzero(output["tau"]) == 0
    assert output["blank_bias"].abs().item() == 0


def test_bound_and_gradient():
    torch.manual_seed(12)
    calibrator = FDGCCalibrator(
        num_classes=6,
        blank_id=0,
        tau_max=2.0,
        delta_max=0.5,
        blank_bias_max=1.0,
    )
    logits = torch.randn(7, 2, 6, requires_grad=True)
    source_prior = torch.tensor([0.6, 0.2, 0.1, 0.07, 0.03])
    target_prior = torch.full_like(source_prior, 0.2)
    difficulty = torch.linspace(0.0, 1.0, 5)
    mask = torch.ones(7, 2, dtype=torch.bool)
    targets = torch.tensor(
        [
            [1, 2],
            [0, 1],
            [2, 0],
            [3, 4],
            [0, 5],
            [1, 0],
            [2, 3],
        ]
    )

    output = calibrator(
        logits,
        source_prior,
        target_prior,
        difficulty=difficulty,
        frame_mask=mask,
    )
    loss = calibrator.calibration_loss(output, targets, mask)
    loss.backward()

    assert output["tau"].min().item() >= 0
    assert output["tau"].max().item() <= 2.0
    assert output["blank_bias"].abs().item() <= 1.0
    assert calibrator.class_mlp[-1].weight.grad is not None
    assert calibrator.frame_mlp[-1].weight.grad is not None
    assert calibrator.class_mlp[-1].weight.grad.abs().sum().item() > 0
    assert calibrator.frame_mlp[-1].weight.grad.abs().sum().item() > 0


def test_blank_index_support():
    calibrator = FDGCCalibrator(num_classes=4, blank_id=3)
    logits = torch.randn(5, 1, 4)
    prior = torch.tensor([0.5, 0.3, 0.2])
    output = calibrator(logits, prior)
    torch.testing.assert_close(output["logits"], logits, atol=1e-6, rtol=1e-6)


def test_frame_margin_is_probability_margin_and_shift_invariant():
    logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]])
    probabilities = F.softmax(logits, dim=-1)
    expected = (
        probabilities[..., 0] - probabilities[..., 1]
    )
    actual = _top_margin(logits)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        _top_margin(logits + 17.0),
        actual,
    )
    assert actual.min().item() >= 0.0
    assert actual.max().item() <= 1.0


def test_frequency_and_difficulty_ablations_are_independent():
    logits = torch.randn(4, 1, 5)
    target_prior = torch.full((4,), 0.25)
    prior_a = torch.tensor([0.7, 0.1, 0.1, 0.1])
    prior_b = torch.tensor([0.1, 0.7, 0.1, 0.1])
    difficulty_a = torch.zeros(4)
    difficulty_b = torch.ones(4)

    frequency_off = FDGCCalibrator(
        num_classes=5,
        use_frequency=False,
        use_difficulty=True,
    )
    with torch.no_grad():
        frequency_off.class_mlp[-1].weight.zero_()
        frequency_off.class_mlp[-1].weight[0, 0] = 1.0
        frequency_off.class_mlp[-1].weight[0, 1] = 0.5
    first = frequency_off(
        logits,
        prior_a,
        target_prior,
        difficulty=difficulty_a,
    )
    second = frequency_off(
        logits,
        prior_b,
        target_prior,
        difficulty=difficulty_a,
    )
    torch.testing.assert_close(first["logits"], second["logits"])
    torch.testing.assert_close(first["tau_class"], second["tau_class"])

    difficulty_off = FDGCCalibrator(
        num_classes=5,
        use_frequency=True,
        use_difficulty=False,
    )
    with torch.no_grad():
        difficulty_off.class_mlp[-1].weight.zero_()
        difficulty_off.class_mlp[-1].weight[0, 0] = 1.0
        difficulty_off.class_mlp[-1].weight[0, 1] = 0.5
    first = difficulty_off(
        logits,
        prior_a,
        target_prior,
        difficulty=difficulty_a,
    )
    second = difficulty_off(
        logits,
        prior_a,
        target_prior,
        difficulty=difficulty_b,
    )
    torch.testing.assert_close(first["logits"], second["logits"])
    torch.testing.assert_close(first["tau_class"], second["tau_class"])


def test_calibrator_accepts_log_probabilities_explicitly():
    calibrator = FDGCCalibrator(num_classes=5)
    logits = torch.randn(4, 1, 5)
    log_probs = logits.log_softmax(dim=-1)
    output = calibrator(
        log_probs,
        torch.full((4,), 0.25),
        input_is_log_probs=True,
    )
    torch.testing.assert_close(output["logits"], log_probs)
    torch.testing.assert_close(
        output["log_probs"].exp().sum(dim=-1),
        torch.ones(4, 1),
    )


if __name__ == "__main__":
    test_identity_initialization_and_normalization()
    test_bound_and_gradient()
    test_blank_index_support()
    test_frame_margin_is_probability_margin_and_shift_invariant()
    test_frequency_and_difficulty_ablations_are_independent()
    test_calibrator_accepts_log_probabilities_explicitly()
    print("calibration tests passed")
