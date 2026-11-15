import torch

from modules.weighted_ctc import (
    class_weights_from_counts,
    class_weights_from_prior,
    frequency_weights,
    weighted_emission_log_probs,
)


def test_frequency_weights_are_bounded_and_point_tail_upward():
    counts = torch.tensor([1000.0, 100.0, 10.0, 1.0])
    weights = frequency_weights(counts, max_weight=2.0, mapping="linear")
    assert torch.all(weights >= 1.0)
    assert torch.all(weights <= 2.0)
    assert weights[0].item() == 1.0
    assert weights[-1].item() == 2.0
    assert torch.all(weights[1:] > weights[:-1])


def test_sigmoid_mapping_preserves_endpoints():
    counts = torch.tensor([1000.0, 100.0, 10.0, 1.0])
    weights = frequency_weights(
        counts,
        max_weight=3.0,
        temperature=1.5,
        mapping="sigmoid",
    )
    torch.testing.assert_close(weights[0], torch.tensor(1.0))
    torch.testing.assert_close(weights[-1], torch.tensor(3.0))


def test_equal_frequency_has_no_artificial_reweighting():
    weights = frequency_weights(torch.ones(5), max_weight=4.0)
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_full_weight_vector_keeps_blank_separate():
    counts = torch.tensor([100.0, 10.0, 1.0])
    weights = class_weights_from_counts(
        counts,
        max_weight=2.5,
        blank_weight=1.0,
    )
    assert weights.shape == (4,)
    assert weights[0].item() == 1.0
    assert weights[-1].item() == 2.5

    prior = torch.tensor([0.0, 100.0, 10.0, 1.0])
    from_prior = class_weights_from_prior(prior, max_weight=2.5)
    torch.testing.assert_close(weights, from_prior)


def test_weighted_emissions_are_normalized_and_identity_for_unit_weights():
    logits = torch.randn(7, 2, 5, requires_grad=True)
    baseline = logits.log_softmax(dim=-1)
    identity = weighted_emission_log_probs(logits, torch.ones(5))
    torch.testing.assert_close(identity, baseline, atol=0.0, rtol=0.0)
    probabilities = identity.exp()
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.isfinite(identity).all()


def test_invalid_weight_inputs_are_rejected():
    logits = torch.randn(3, 1, 4)
    for weights in (
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 0.0, 1.0]),
        torch.tensor([1.0, 1.0, float("nan"), 1.0]),
    ):
        try:
            weighted_emission_log_probs(logits, weights)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid class weights were accepted")


if __name__ == "__main__":
    test_frequency_weights_are_bounded_and_point_tail_upward()
    test_sigmoid_mapping_preserves_endpoints()
    test_equal_frequency_has_no_artificial_reweighting()
    test_full_weight_vector_keeps_blank_separate()
    test_weighted_emissions_are_normalized_and_identity_for_unit_weights()
    test_invalid_weight_inputs_are_rejected()
    print("weighted CTC tests passed")
