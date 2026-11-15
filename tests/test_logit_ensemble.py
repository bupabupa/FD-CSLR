import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_logit_ensemble import (
    apply_class_prior_adjustment,
    combine_probabilities,
    find_test_access_claim,
    load_model,
    load_ensemble_spec,
    parse_member,
    validate_members,
)


def _member(root, name, weight=1.0, temperature=1.0):
    config = root / "{}.yaml".format(name)
    checkpoint = root / "{}.pt".format(name)
    config.write_text("dataset: phoenix14\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    return parse_member(
        "name={};config={};checkpoint={};weight={};temperature={}".format(
            name,
            config,
            checkpoint,
            weight,
            temperature,
        )
    )


def test_parse_and_validate_members():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        members = [
            _member(root, "first", weight=2.0, temperature=0.5),
            _member(root, "second", weight=1.0),
        ]
        validate_members(members)
    assert members[0]["weight"] == 2.0
    assert members[0]["temperature"] == 0.5


def test_weighted_probability_combination():
    first = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    second = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
    members = [{"weight": 1.0, "temperature": 1.0, "name": "a"},
               {"weight": 1.0, "temperature": 1.0, "name": "b"}]
    combined = combine_probabilities([first, second], members)
    assert torch.allclose(
        combined.sum(dim=-1),
        torch.ones_like(combined[..., 0]),
    )
    assert torch.allclose(combined, torch.full_like(combined, 0.5))


def test_rare_class_prior_boost_is_normalized():
    probabilities = torch.tensor([[[0.7, 0.29, 0.01]]])
    prior = torch.tensor([0.0, 0.9, 0.1])
    adjusted = apply_class_prior_adjustment(probabilities, prior, 0.5)
    assert torch.allclose(
        adjusted.sum(dim=-1),
        torch.ones_like(adjusted[..., 0]),
    )
    assert adjusted[..., 2] > probabilities[..., 2]
    assert (
        adjusted[..., 1] / adjusted[..., 2]
        < probabilities[..., 1] / probabilities[..., 2]
    )
    assert torch.equal(
        apply_class_prior_adjustment(probabilities, prior, 0.0),
        probabilities,
    )


def test_test_claim_is_found_in_ancestor():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        claim = {
            "selected": {
                "name": "candidate",
                "dev_prediction": "dev.jsonl",
                "config": "config.yaml",
                "checkpoint": "checkpoint.pt",
            }
        }
        (root / "test_access_claim.json").write_text(
            json.dumps(claim),
            encoding="utf-8",
        )
        output = root / "locked_test" / "candidate"
        output.mkdir(parents=True)
        assert find_test_access_claim(output) == root / "test_access_claim.json"


def test_ensemble_spec_paths_are_relative_to_spec():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        configs = root / "configs"
        checkpoints = root / "checkpoints"
        configs.mkdir()
        checkpoints.mkdir()
        (configs / "member.yaml").write_text(
            "dataset: phoenix14\n",
            encoding="utf-8",
        )
        (checkpoints / "member.pt").write_bytes(b"checkpoint")
        np.save(root / "prior.npy", np.array([0.0, 1.0], dtype=np.float32))
        spec_path = root / "ensemble.json"
        spec_path.write_text(
            json.dumps({
                "members": [{
                    "name": "member",
                    "config": "configs/member.yaml",
                    "checkpoint": "checkpoints/member.pt",
                }],
                "class_prior": {
                    "path": "prior.npy",
                    "tau": 0.1,
                },
            }),
            encoding="utf-8",
        )
        spec = load_ensemble_spec(spec_path)
    assert spec["members"][0]["config"].name == "member.yaml"
    assert spec["class_prior_path"].name == "prior.npy"


def test_ensemble_loader_rejects_legacy_fusion_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "legacy.pt"
        torch.save(
            {
                "model_state_dict": {
                    "multichannel_calibrator.fusion_logits": torch.ones(3),
                },
            },
            checkpoint,
        )
        member = {"checkpoint": checkpoint}
        args = SimpleNamespace(split_manifest=None)
        try:
            load_model(
                member,
                args,
                gloss_dict={},
                device="cpu",
                allow_unbound_checkpoints=False,
            )
        except ValueError as error:
            assert "metadata version" in str(error)
            return
    raise AssertionError(
        "logit ensemble loader accepted a legacy fusion checkpoint"
    )


if __name__ == "__main__":
    test_parse_and_validate_members()
    test_weighted_probability_combination()
    test_rare_class_prior_boost_is_normalized()
    test_test_claim_is_found_in_ancestor()
    test_ensemble_spec_paths_are_relative_to_spec()
    test_ensemble_loader_rejects_legacy_fusion_checkpoint()
    print("logit ensemble tests passed")
