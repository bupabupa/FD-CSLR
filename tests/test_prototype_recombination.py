import math

import torch

from modules.prototype_recombination import PrototypeRecombination


def _inputs():
    torch.manual_seed(0)
    features = torch.randn(5, 2, 4)
    prototypes = torch.randn(4, 4)
    support = torch.tensor([10.0, 1.0, 1.0, 0.0])
    gamma = torch.softmax(torch.randn(5, 2, 4), dim=-1)
    confidence = torch.ones(5, 2)
    return features, prototypes, support, gamma, confidence


def test_zero_initialized_strength_is_exact_identity():
    features, prototypes, support, gamma, confidence = _inputs()
    module = PrototypeRecombination(max_strength=0.25)
    output = module(features, prototypes, support, gamma, confidence)
    torch.testing.assert_close(output["features"], features)
    assert output["strength"].item() == 0.0
    assert output["mean_shift"].item() == 0.0


def test_strength_is_bounded_and_receives_gradient():
    features, prototypes, support, gamma, confidence = _inputs()
    module = PrototypeRecombination(max_strength=0.25)
    with torch.no_grad():
        module.strength_raw.fill_(math.atanh(0.5))
    output = module(features, prototypes, support, gamma, confidence)
    assert 0.0 < output["strength"].item() < 0.25
    output["mean_shift"].backward()
    assert module.strength_raw.grad is not None
    assert torch.isfinite(module.strength_raw.grad)
    assert module.strength_raw.grad.abs().item() > 0


def test_unsupported_and_blank_classes_do_not_enter_the_mix():
    features = torch.zeros(1, 1, 2)
    prototypes = torch.tensor(
        [
            [100.0, 100.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [50.0, 50.0],
        ]
    )
    support = torch.tensor([100.0, 1.0, 0.0, 1.0])
    gamma = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])
    module = PrototypeRecombination(max_strength=0.25, min_mix_mass=0.1)
    with torch.no_grad():
        module.strength_raw.fill_(math.atanh(0.5))
    output = module(
        features,
        prototypes,
        support,
        gamma,
        torch.ones(1, 1),
    )
    torch.testing.assert_close(output["prototype_mix"], prototypes[1].view(1, 1, 2))
    assert output["active"].all()


def test_low_posterior_mass_or_confidence_deactivates_frames():
    features = torch.zeros(2, 1, 2)
    prototypes = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    support = torch.tensor([10.0, 1.0])
    gamma = torch.tensor([[[1.0, 0.0]], [[0.9, 0.1]]])
    confidence = torch.tensor([[1.0], [0.0]])
    module = PrototypeRecombination(
        max_strength=0.25,
        min_mix_mass=0.5,
        confidence_threshold=0.5,
    )
    with torch.no_grad():
        module.strength_raw.fill_(math.atanh(0.5))
    output = module(features, prototypes, support, gamma, confidence)
    assert output["active"].tolist() == [[False], [False]]
    torch.testing.assert_close(output["features"], features)


if __name__ == "__main__":
    test_zero_initialized_strength_is_exact_identity()
    test_strength_is_bounded_and_receives_gradient()
    test_unsupported_and_blank_classes_do_not_enter_the_mix()
    test_low_posterior_mass_or_confidence_deactivates_frames()
    print("prototype recombination tests passed")
