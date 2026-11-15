import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset.splits import load_split_manifest
from scripts.evaluate_logit_ensemble import (
    file_sha256,
    load_ensemble_spec,
    load_yaml,
)


def load_checkpoint(path):
    # Training checkpoints include trusted NumPy RNG state, which the
    # restricted weights-only unpickler cannot reconstruct.
    payload = torch.load(
        str(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary: {}".format(
            path
        ))
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("checkpoint has no model_state_dict: {}".format(path))
    return payload


def validate_split_hashes(spec, member_payloads, split_sha256):
    if spec.get("split_sha256") not in (None, split_sha256):
        raise ValueError("ensemble specification split hash mismatch")
    for member in spec["members"]:
        payload = member_payloads[member["name"]]
        if payload.get("split_sha256") != split_sha256:
            raise ValueError(
                "member checkpoint is not bound to the strict split: {}".format(
                    member["checkpoint"]
                )
            )


def normalize_member_model_args(model_args):
    model_args = copy.deepcopy(model_args)
    for key in ("class_prior_path", "rescorer_path"):
        value = model_args.get(key)
        if value:
            model_args[key] = str(Path(value).expanduser().resolve())
    return model_args


def build_ensemble_state(
    members,
    member_payloads,
    member_weights,
    member_temperatures,
    class_prior,
    prior_tau,
):
    if len(members) != len(member_weights) or len(members) != len(
        member_temperatures
    ):
        raise ValueError("ensemble member metadata is not aligned")
    state = {
        "member_weights": torch.tensor(
            member_weights,
            dtype=torch.float32,
        ),
        "member_temperatures": torch.tensor(
            member_temperatures,
            dtype=torch.float32,
        ),
        "class_prior": torch.as_tensor(
            class_prior,
            dtype=torch.float32,
        ).reshape(-1),
        "prior_tau": torch.tensor(float(prior_tau), dtype=torch.float32),
    }
    for index, member in enumerate(members):
        member_state = member_payloads[member["name"]]["model_state_dict"]
        prefix = "members.member_{}.".format(index)
        for key, value in member_state.items():
            state[prefix + key] = value
    return state


def build_ensemble_config(spec, split_manifest, num_classes):
    first_config = load_yaml(spec["members"][0]["config"])
    config = copy.deepcopy(first_config)
    member_specs = []
    for member in spec["members"]:
        member_config = load_yaml(member["config"])
        member_specs.append({
            "name": member["name"],
            "model": member_config["model"],
            "model_args": normalize_member_model_args(
                member_config.get("model_args", {})
            ),
            "loss_weights": copy.deepcopy(
                member_config.get("loss_weights", {})
            ),
            "weight": float(member["weight"]),
            "temperature": float(member["temperature"]),
        })

    first_model_args = first_config.get("model_args", {})
    config["phase"] = "test"
    config["model"] = "ensemble_network.WeightedEnsembleModel"
    config["split_manifest"] = str(split_manifest)
    config["model_args"] = {
        "members": member_specs,
        "beam_width": int(first_model_args.get("beam_width", 10)),
        "num_processes": int(first_model_args.get("num_processes", 10)),
        "class_prior_path": (
            str(spec["class_prior_path"])
            if spec.get("class_prior_path") is not None
            else None
        ),
        "prior_tau": float(spec.get("prior_tau", 0.0)),
        "num_classes": int(num_classes),
    }
    config["loss_weights"] = copy.deepcopy(
        first_config.get("loss_weights", {})
    )
    return config


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def bundle(args):
    spec_path = Path(args.spec).expanduser().resolve()
    split_manifest = Path(args.split_manifest).expanduser().resolve()
    spec = load_ensemble_spec(spec_path)
    split_info = load_split_manifest(
        str(split_manifest),
        os.path.dirname(str(split_manifest)),
    )
    split_sha256 = split_info["split_sha256"]
    member_payloads = {
        member["name"]: load_checkpoint(member["checkpoint"])
        for member in spec["members"]
    }
    if not args.allow_unbound_checkpoints:
        validate_split_hashes(spec, member_payloads, split_sha256)

    first_config = load_yaml(spec["members"][0]["config"])
    root = Path(__file__).resolve().parents[1]
    dataset_info = load_yaml(
        root / "configs" / "{}.yaml".format(first_config["dataset"])
    )
    gloss_dict = np.load(
        dataset_info["dict_path"],
        allow_pickle=True,
    ).item()
    num_classes = len(gloss_dict) + 1
    if args.class_prior_path is not None:
        class_prior = np.asarray(
            np.load(args.class_prior_path),
            dtype=np.float32,
        ).reshape(-1)
    elif spec.get("class_prior_path") is not None:
        class_prior = np.asarray(
            np.load(spec["class_prior_path"]),
            dtype=np.float32,
        ).reshape(-1)
    else:
        class_prior = torch.zeros(num_classes, dtype=torch.float32)
    if tuple(class_prior.shape) != (num_classes,):
        raise ValueError("class prior shape does not match strict num_classes")
    if not torch.isfinite(torch.as_tensor(class_prior)).all():
        raise ValueError("class prior contains non-finite values")
    if (torch.as_tensor(class_prior) < 0.0).any():
        raise ValueError("class prior contains negative values")

    prior_tau = (
        float(args.prior_tau)
        if args.prior_tau is not None
        else float(spec.get("prior_tau", 0.0))
    )
    if prior_tau < 0.0:
        raise ValueError("prior tau must be non-negative")
    weights = [member["weight"] for member in spec["members"]]
    temperatures = [
        member["temperature"] for member in spec["members"]
    ]
    state = build_ensemble_state(
        spec["members"],
        member_payloads,
        weights,
        temperatures,
        class_prior,
        prior_tau,
    )
    config = build_ensemble_config(spec, split_manifest, num_classes)
    output_checkpoint = Path(args.output_checkpoint).expanduser().resolve()
    output_config = Path(args.output_config).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config, "w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination, sort_keys=False)

    metadata = {
        "format": "weighted_ctc_probability_ensemble_v1",
        "spec_path": str(spec_path),
        "spec_sha256": file_sha256(spec_path),
        "split_manifest": str(split_manifest),
        "split_sha256": split_sha256,
        "members": [
            {
                **{
                    key: (
                        str(value)
                        if isinstance(value, Path)
                        else value
                    )
                    for key, value in member.items()
                },
                "checkpoint_sha256": file_sha256(member["checkpoint"]),
                "config_sha256": file_sha256(member["config"]),
            }
            for member in spec["members"]
        ],
        "class_prior": (
            {
                "path": str(Path(args.class_prior_path).resolve()),
                "sha256": file_sha256(args.class_prior_path),
            }
            if args.class_prior_path is not None
            else (
                {
                    "path": str(spec["class_prior_path"]),
                    "sha256": file_sha256(spec["class_prior_path"]),
                }
                if spec.get("class_prior_path") is not None
                else None
            )
        ),
        "prior_tau": prior_tau,
        "diagnostic_unbound_checkpoints": bool(
            args.allow_unbound_checkpoints
        ),
        "config": str(output_config),
        "config_sha256": file_sha256(output_config),
    }
    atomic_torch_save({
        "epoch": int(args.epoch),
        "model_state_dict": state,
        "split_sha256": split_sha256,
        "ensemble_metadata": metadata,
    }, output_checkpoint)
    metadata["checkpoint"] = str(output_checkpoint)
    metadata["checkpoint_sha256"] = file_sha256(output_checkpoint)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as destination:
        json.dump(metadata, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print("wrote ensemble checkpoint: {}".format(output_checkpoint))
    print("wrote ensemble config: {}".format(output_config))
    print("wrote ensemble manifest: {}".format(output_manifest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--class-prior-path", default=None)
    parser.add_argument("--prior-tau", type=float, default=None)
    parser.add_argument("--epoch", type=int, default=6667)
    parser.add_argument("--allow-unbound-checkpoints", action="store_true")
    args = parser.parse_args()
    bundle(args)


if __name__ == "__main__":
    main()
