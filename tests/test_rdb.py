import math

import torch

from fd_xmp_network import FDXMPModel
from modules.rdb import RDB


def _inputs():
    torch.manual_seed(7)
    features = torch.randn(6, 2, 4, requires_grad=True)
    prototypes = torch.randn(5, 4)
    support = torch.tensor([10.0, 2.0, 2.0, 1.0, 1.0])
    gamma = torch.softmax(torch.randn(6, 2, 5), dim=-1)
    confidence = torch.linspace(0.2, 1.0, 12).view(6, 2)
    lengths = torch.tensor([6, 4])
    return features, prototypes, support, gamma, confidence, lengths


def test_zero_initialized_output_is_exact_identity():
    features, prototypes, support, gamma, confidence, lengths = _inputs()
    module = RDB(feature_dim=4)
    output = module(
        features,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=lengths,
    )
    torch.testing.assert_close(output["features"], features)
    assert output["mean_shift"].item() == 0.0
    assert torch.count_nonzero(module.output.weight) == 0
    assert torch.count_nonzero(module.output.bias) == 0


def test_strength_is_bounded_and_receives_gradient():
    features, prototypes, support, gamma, confidence, lengths = _inputs()
    module = RDB(feature_dim=4, max_strength=0.2)
    output = module(
        features,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=lengths,
    )
    assert 0.0 < output["strength"].item() < 0.2
    with torch.no_grad():
        module.output.weight.normal_(0.0, 0.05)
    output = module(
        features,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=lengths,
    )
    output["features"].sum().backward()
    assert module.strength_raw.grad is not None
    assert torch.isfinite(module.strength_raw.grad)
    assert module.strength_raw.grad.abs().item() > 0


def test_padding_does_not_enter_reconstruction_loss():
    features, prototypes, support, gamma, confidence, lengths = _inputs()
    module = RDB(feature_dim=4)
    output = module(
        features,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=lengths,
    )
    reconstructed = features.detach().clone()
    reconstructed[4:, 1] = 1000.0
    loss_with_padding = module.reconstruction_loss(
        features.detach(),
        reconstructed,
        lengths=lengths,
        active=output["active"],
    )["loss"]
    reconstructed[4:, 1] = 0.0
    loss_without_padding = module.reconstruction_loss(
        features.detach(),
        reconstructed,
        lengths=lengths,
        active=output["active"],
    )["loss"]
    torch.testing.assert_close(loss_with_padding, loss_without_padding)
    padding_output = output["features"][4:, 1]
    torch.testing.assert_close(padding_output, features[4:, 1])


def test_padding_does_not_change_valid_reconstruction():
    features, prototypes, support, gamma, confidence, lengths = _inputs()
    first = RDB(feature_dim=4)
    second = RDB(feature_dim=4)
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        first.output.weight.normal_(0.0, 0.05)
        second.load_state_dict(first.state_dict())
        changed_padding = features.detach().clone()
        changed_padding[4:, 1] = 100.0
    first_output = first(
        features.detach(),
        prototypes,
        support,
        gamma,
        confidence,
        lengths=lengths,
    )
    second_output = second(
        changed_padding,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=lengths,
    )
    torch.testing.assert_close(
        first_output["features"][:4, 1],
        second_output["features"][:4, 1],
    )


def test_unsupported_classes_do_not_enter_content_mix():
    features = torch.zeros(1, 1, 2)
    prototypes = torch.tensor(
        [
            [100.0, 100.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [50.0, 50.0],
            [25.0, 25.0],
        ]
    )
    support = torch.tensor([100.0, 1.0, 0.0, 1.0, 1.0])
    gamma = torch.tensor([[[0.0, 0.8, 0.2, 0.0, 0.0]]])
    confidence = torch.ones(1, 1)
    module = RDB(feature_dim=2)
    output = module(
        features,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=torch.tensor([1]),
    )
    torch.testing.assert_close(
        output["content_mix"],
        prototypes[1].view(1, 1, 2),
    )
    assert output["active"].all()


def test_signer_style_swap_is_bounded_and_differentiable():
    features, prototypes, support, gamma, confidence, lengths = _inputs()
    module = RDB(feature_dim=4, max_strength=0.2)
    with torch.no_grad():
        module.output.weight.normal_(0.0, 0.05)
    style = torch.randn(2, 8, requires_grad=True)
    counterfactual_style = torch.randn(2, 8, requires_grad=True)
    output = module(
        features,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=lengths,
        style=style,
        counterfactual_style=counterfactual_style,
        style_swap_mask=torch.tensor([True, True]),
    )
    difference = (
        output["counterfactual_features"] - output["features"]
    ).detach()
    assert output["counterfactual_features"].requires_grad
    assert difference.abs().max().item() <= 2.0 * 0.2 + 1e-6
    (
        output["features"].sum()
        + output["counterfactual_features"].sum()
    ).backward()
    assert style.grad is not None
    assert counterfactual_style.grad is not None
    assert torch.isfinite(style.grad).all()
    assert torch.isfinite(counterfactual_style.grad).all()
    assert style.grad.abs().sum().item() > 0
    assert counterfactual_style.grad.abs().sum().item() > 0


def test_output_shape_is_time_preserving():
    features, prototypes, support, gamma, confidence, lengths = _inputs()
    module = RDB(feature_dim=4)
    output = module(
        features,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=lengths,
    )
    assert output["features"].shape == features.shape
    assert output["temporal_residual"].shape == features.shape


def test_nonfinite_inputs_and_bad_shapes_raise():
    features, prototypes, support, gamma, confidence, lengths = _inputs()
    module = RDB(feature_dim=4)
    bad_features = features.detach().clone()
    bad_features[0, 0, 0] = float("nan")
    try:
        module(
            bad_features,
            prototypes,
            support,
            gamma,
            confidence,
            lengths=lengths,
        )
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite features were accepted")

    try:
        module(
            features,
            prototypes,
            support,
            gamma[:, :, :-1],
            confidence,
            lengths=lengths,
        )
    except ValueError as error:
        assert "gamma" in str(error)
    else:
        raise AssertionError("incorrect gamma shape was accepted")

    try:
        RDB(feature_dim=4, kernel_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("even temporal kernel was accepted")


class _CriterionHarness:
    criterion_calculation = FDXMPModel.criterion_calculation
    _select_label_sequences = staticmethod(
        FDXMPModel._select_label_sequences
    )

    def __init__(self):
        self.loss_weights = {
            "RDBRecon": 0.05,
            "RDBCTC": 0.05,
            "RDBStyle": 0.02,
        }
        self.rdb_style_separation_weight = 1.0
        self._training_epoch = None
        self.rdb_recon_warmup_epochs = 5
        self.calls = []

    def _alignment_warmup_factor(self, name):
        return FDXMPModel._alignment_warmup_factor(self, name)

    def _ctc_branch_loss(
        self,
        logits,
        label,
        input_lengths,
        target_lengths,
        branch_name,
        **kwargs
    ):
        self.calls.append(
            (
                branch_name,
                label.detach().clone(),
                target_lengths.detach().clone(),
            )
        )
        return logits.sum() * 0.0 + 1.0


def test_criterion_reuses_original_labels_for_rdb_ctc_branches():
    harness = _CriterionHarness()
    sequence_logits = torch.zeros(4, 2, 5)
    label = torch.tensor([2, 3])
    label_lgt = torch.tensor([1, 1])
    ret_dict = {
        "sequence_logits": sequence_logits,
        "rdb_recon_loss": sequence_logits.sum() * 0.0 + 0.25,
        "rdb_logits": torch.zeros(4, 2, 5),
        "rdb_style_logits": torch.zeros(4, 1, 5),
        "rdb_style_feat_len": torch.tensor([4]),
        "rdb_style_swap_mask": torch.tensor([True, False]),
        "rdb_style_separation_loss": sequence_logits.sum() * 0.0 + 0.5,
        "feat_len": torch.tensor([4, 3]),
    }
    loss = harness.criterion_calculation(ret_dict, label, label_lgt)
    assert torch.isfinite(loss)
    calls = {name: (labels, lengths) for name, labels, lengths in harness.calls}
    torch.testing.assert_close(calls["RDBCTC"][0], label)
    torch.testing.assert_close(calls["RDBCTC"][1], label_lgt)
    torch.testing.assert_close(
        calls["RDBStyle"][0], torch.tensor([2])
    )
    torch.testing.assert_close(
        calls["RDBStyle"][1], torch.tensor([1])
    )
    assert harness.last_rdb_style_available


if __name__ == "__main__":
    test_zero_initialized_output_is_exact_identity()
    test_strength_is_bounded_and_receives_gradient()
    test_padding_does_not_enter_reconstruction_loss()
    test_padding_does_not_change_valid_reconstruction()
    test_unsupported_classes_do_not_enter_content_mix()
    test_signer_style_swap_is_bounded_and_differentiable()
    test_output_shape_is_time_preserving()
    test_nonfinite_inputs_and_bad_shapes_raise()
    test_criterion_reuses_original_labels_for_rdb_ctc_branches()
    print("RDB tests passed")
