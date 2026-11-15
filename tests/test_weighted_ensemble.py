from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn as nn

import ensemble_network
from ensemble_network import (
    WeightedEnsembleModel,
    class_prior_adjustment,
    weighted_probability_average,
)
from scripts.bundle_weighted_ensemble import (
    build_ensemble_state,
    load_checkpoint,
)
from scripts.evaluate_logit_ensemble import (
    apply_class_prior_adjustment as evaluator_prior_adjustment,
    combine_probabilities,
)
from scripts.select_weighted_ensemble import (
    deterministic_anchor_specs,
    ensure_candidate,
    normalize_weights,
    subset_candidates,
)


class ToyMember(nn.Module):
    def __init__(
        self,
        num_classes,
        gloss_dict=None,
        loss_weights=None,
        bias=None,
        **unused
    ):
        super().__init__()
        self.supports_signer = False
        if bias is None:
            bias = [0.0] * num_classes
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

    def forward(
        self,
        x,
        len_x,
        label=None,
        label_lgt=None,
        decode=True,
        signer_ids=None,
    ):
        logits = x + self.bias.view(1, 1, -1)
        return {
            "framewise_features": x,
            "visual_features": x,
            "feat_len": len_x,
            "conv_logits": logits,
            "sequence_logits": logits,
            "conv_sents": None,
            "recognized_sents": None,
        }


def test_weighted_probability_average_is_normalized():
    first = torch.tensor([[[4.0, 0.0, 0.0]]])
    second = torch.tensor([[[0.0, 4.0, 0.0]]])
    combined = weighted_probability_average(
        [first, second],
        [3.0, 1.0],
        [1.0, 1.0],
    )
    assert torch.allclose(
        combined.sum(dim=-1),
        torch.ones_like(combined[..., 0]),
    )
    assert combined[0, 0, 0] > combined[0, 0, 1]


def test_class_prior_adjustment_boosts_rare_classes():
    probabilities = torch.tensor([[[0.7, 0.29, 0.01]]])
    prior = torch.tensor([0.0, 0.9, 0.1])
    adjusted = class_prior_adjustment(probabilities, prior, 0.5)
    assert torch.allclose(
        adjusted.sum(dim=-1),
        torch.ones_like(adjusted[..., 0]),
    )
    assert adjusted[..., 2] > probabilities[..., 2]


def test_wrapper_averages_members_and_embeds_prior():
    original_import = ensemble_network.import_class
    ensemble_network.import_class = lambda name: ToyMember
    try:
        with tempfile.TemporaryDirectory() as directory:
            prior_path = Path(directory) / "prior.npy"
            np.save(prior_path, np.array([0.0, 0.8, 0.2], dtype=np.float32))
            model = WeightedEnsembleModel(
                num_classes=3,
                members=[
                    {
                        "name": "first",
                        "model": "toy.Member",
                        "model_args": {"bias": [2.0, 0.0, 0.0]},
                    },
                    {
                        "name": "second",
                        "model": "toy.Member",
                        "model_args": {"bias": [0.0, 2.0, 0.0]},
                    },
                ],
                gloss_dict={"A": (1, 0), "B": (2, 0)},
                class_prior_path=str(prior_path),
                prior_tau=0.5,
            )
    finally:
        ensemble_network.import_class = original_import
    model.eval()
    output = model(
        torch.zeros(2, 1, 3),
        torch.tensor([2]),
        decode=False,
    )
    assert output["recognized_sents"] is None
    probabilities = output["sequence_logits"].exp()
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
    )
    assert model.member_names == ["first", "second"]
    first = torch.zeros(2, 1, 3) + torch.tensor([2.0, 0.0, 0.0]).view(1, 1, -1)
    second = torch.zeros(2, 1, 3) + torch.tensor([0.0, 2.0, 0.0]).view(1, 1, -1)
    evaluator_probabilities = combine_probabilities(
        [first, second],
        [
            {"name": "first", "weight": 1.0, "temperature": 1.0},
            {"name": "second", "weight": 1.0, "temperature": 1.0},
        ],
    )
    evaluator_probabilities = evaluator_prior_adjustment(
        evaluator_probabilities,
        np.array([0.0, 0.8, 0.2], dtype=np.float32),
        0.5,
    )
    assert torch.allclose(
        probabilities,
        evaluator_probabilities,
        atol=1e-6,
    )


def test_bundle_state_prefixes_member_parameters():
    members = [{"name": "first"}, {"name": "second"}]
    payloads = {
        "first": {
            "model_state_dict": {
                "bias": torch.tensor([1.0, 0.0, 0.0]),
            },
        },
        "second": {
            "model_state_dict": {
                "bias": torch.tensor([0.0, 2.0, 0.0]),
            },
        },
    }
    state = build_ensemble_state(
        members,
        payloads,
        [0.25, 0.75],
        [1.0, 2.0],
        np.array([0.0, 0.5, 0.5], dtype=np.float32),
        0.1,
    )
    assert torch.equal(
        state["members.member_0.bias"],
        torch.tensor([1.0, 0.0, 0.0]),
    )
    assert torch.equal(
        state["members.member_1.bias"],
        torch.tensor([0.0, 2.0, 0.0]),
    )
    assert torch.allclose(
        state["member_weights"],
        torch.tensor([0.25, 0.75]),
    )
    original_import = ensemble_network.import_class
    ensemble_network.import_class = lambda name: ToyMember
    try:
        model = WeightedEnsembleModel(
            num_classes=3,
            members=[
                {
                    "name": "first",
                    "model": "toy.Member",
                    "model_args": {"bias": [0.0, 0.0, 0.0]},
                    "weight": 0.25,
                    "temperature": 1.0,
                },
                {
                    "name": "second",
                    "model": "toy.Member",
                    "model_args": {"bias": [0.0, 0.0, 0.0]},
                    "weight": 0.75,
                    "temperature": 2.0,
                },
            ],
            gloss_dict={"A": (1, 0), "B": (2, 0)},
        )
    finally:
        ensemble_network.import_class = original_import
    model.load_state_dict(state, strict=True)
    assert torch.equal(
        model.members["member_0"].bias,
        torch.tensor([1.0, 0.0, 0.0]),
    )


def test_bundle_loader_accepts_training_rng_state():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save({
            "model_state_dict": {
                "bias": torch.tensor([1.0, 0.0, 0.0]),
            },
            "rng_state": {
                "numpy": np.random.RandomState(0).get_state(),
            },
        }, checkpoint)
        payload = load_checkpoint(checkpoint)
    assert torch.equal(
        payload["model_state_dict"]["bias"],
        torch.tensor([1.0, 0.0, 0.0]),
    )


def test_tuning_helpers_cover_subsets_and_normalize_weights():
    subsets = list(subset_candidates(4, 2))
    assert subsets[0] == (0,)
    assert (0, 1) in subsets
    assert (0, 1, 2) not in subsets
    weights = normalize_weights([2.0, 1.0, 1.0])
    assert np.allclose(weights, np.array([0.5, 0.25, 0.25]))
    summary = []
    ensure_candidate(summary, {
        "member_indices": (0, 1),
        "weights": (0.5, 0.5),
        "temperatures": (1.0, 1.0),
        "prior_tau": 0.05,
        "cal_nll": 12.0,
    })
    assert summary[0]["member_indices"] == [0, 1]
    anchors = list(deterministic_anchor_specs((0, 1)))
    assert any(
        np.allclose(weights, (0.9, 0.1))
        and np.allclose(temperatures, (0.75, 0.9))
        and tau == 0.025
        for weights, temperatures, tau in anchors
    )


if __name__ == "__main__":
    test_weighted_probability_average_is_normalized()
    test_class_prior_adjustment_boosts_rare_classes()
    test_wrapper_averages_members_and_embeds_prior()
    test_bundle_state_prefixes_member_parameters()
    test_bundle_loader_accepts_training_rng_state()
    test_tuning_helpers_cover_subsets_and_normalize_weights()
    print("weighted ensemble tests passed")
