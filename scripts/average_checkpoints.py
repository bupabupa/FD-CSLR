#!/usr/bin/env python3
"""Average compatible model checkpoints without touching evaluation data."""

import argparse
import json
from pathlib import Path
import pickle
import re

import torch


EPOCH_PATTERN = re.compile(r"_epoch([0-9]+)_model\.pt$")


def checkpoint_epoch(path):
    match = EPOCH_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else -1


def discover_checkpoints(run_dir, last_n=None, min_epoch=None, max_epoch=None):
    paths = list(Path(run_dir).glob("dev_*_epoch*_model.pt"))
    paths = [path for path in paths if checkpoint_epoch(path) >= 0]
    paths.sort(key=lambda path: (checkpoint_epoch(path), path.name))
    if min_epoch is not None:
        paths = [path for path in paths if checkpoint_epoch(path) >= min_epoch]
    if max_epoch is not None:
        paths = [path for path in paths if checkpoint_epoch(path) <= max_epoch]
    if last_n is not None:
        if last_n < 1:
            raise ValueError("last_n must be positive")
        paths = paths[-last_n:]
    if not paths:
        raise ValueError("no checkpoints matched the requested range")
    return paths


def _load_payload(path, allow_unbound_checkpoints=False):
    try:
        payload = torch.load(
            str(path),
            map_location="cpu",
            weights_only=True,
        )
    except (pickle.UnpicklingError, RuntimeError):
        if not allow_unbound_checkpoints:
            raise
        payload = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary: {}".format(path))
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("checkpoint has no model_state_dict: {}".format(path))
    # Optimizer and scheduler states are unnecessary for weight averaging and
    # can be several times larger than the model itself.
    return {
        "model_state_dict": payload["model_state_dict"],
        "split_sha256": payload.get("split_sha256"),
        "epoch": payload.get("epoch"),
    }


def average_state_dicts(state_dicts):
    if not state_dicts:
        raise ValueError("at least one state dict is required")
    expected_keys = set(state_dicts[0])
    for index, state_dict in enumerate(state_dicts[1:], start=1):
        keys = set(state_dict)
        if keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(
                "state dict {} has incompatible keys; missing={}, extra={}".format(
                    index, missing, extra
                )
            )

    averaged = {}
    for key in state_dicts[0]:
        tensors = [state_dict[key] for state_dict in state_dicts]
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("state dict entry is not a tensor: {}".format(key))
        first = tensors[0]
        for index, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != first.shape or tensor.dtype != first.dtype:
                raise ValueError(
                    "state dict entry {} differs at checkpoint {}: {} vs {}".format(
                        key,
                        index,
                        (tuple(first.shape), first.dtype),
                        (tuple(tensor.shape), tensor.dtype),
                    )
                )
        if first.is_floating_point():
            value = tensors[0].to(torch.float64)
            for tensor in tensors[1:]:
                value.add_(tensor.to(torch.float64))
            averaged[key] = value.div_(len(tensors)).to(first.dtype)
        else:
            if not all(torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ValueError(
                    "integer or boolean state entry is not identical: {}".format(
                        key
                    )
                )
            averaged[key] = first.clone()
    return averaged


def average_checkpoints(
    checkpoint_paths,
    output_path,
    allow_unbound_checkpoints=False,
):
    paths = [Path(path).resolve() for path in checkpoint_paths]
    payloads = [
        _load_payload(
            path,
            allow_unbound_checkpoints=allow_unbound_checkpoints,
        )
        for path in paths
    ]
    missing_hashes = [
        str(path)
        for path, payload in zip(paths, payloads)
        if not payload.get("split_sha256")
    ]
    if missing_hashes and not allow_unbound_checkpoints:
        raise ValueError(
            "checkpoints are missing split_sha256: {}".format(missing_hashes)
        )
    split_hashes = {
        payload["split_sha256"]
        for payload in payloads
        if payload["split_sha256"]
    }
    if len(split_hashes) != 1:
        if split_hashes or not allow_unbound_checkpoints:
            raise ValueError(
                "checkpoints use different split hashes: {}".format(
                    split_hashes
                )
            )
    state_dict = average_state_dicts(
        [payload["model_state_dict"] for payload in payloads]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "epoch": max(checkpoint_epoch(path) for path in paths),
        "model_state_dict": state_dict,
        "split_sha256": (
            next(iter(split_hashes))
            if split_hashes and not missing_hashes
            else None
        ),
        "diagnostic_unbound_checkpoints": bool(missing_hashes),
        "averaged_checkpoints": [
            {
                "path": str(path),
                "epoch": checkpoint_epoch(path),
                "dev_wer": _checkpoint_dev_wer(path),
            }
            for path in paths
        ],
        "average_rule": (
            "arithmetic mean of floating tensors; integer and boolean "
            "buffers must be identical and are copied unchanged"
        ),
    }
    torch.save(output_payload, output_path)
    return output_payload


def _checkpoint_dev_wer(path):
    match = re.match(r"dev_([0-9]+(?:\.[0-9]+)?)_epoch", Path(path).name)
    return float(match.group(1)) if match else None


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir")
    source.add_argument("--checkpoint", action="append")
    parser.add_argument("--last-n", type=int, default=3)
    parser.add_argument("--min-epoch", type=int)
    parser.add_argument("--max-epoch", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-unbound-checkpoints",
        action="store_true",
        help="diagnostic only: allow checkpoints without split hashes",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_dir:
        if args.last_n is None:
            raise ValueError("--last-n is required with --run-dir")
        checkpoints = discover_checkpoints(
            args.run_dir,
            last_n=args.last_n,
            min_epoch=args.min_epoch,
            max_epoch=args.max_epoch,
        )
    else:
        checkpoints = [Path(path) for path in args.checkpoint]
        checkpoints.sort(key=lambda path: (checkpoint_epoch(path), path.name))
    payload = average_checkpoints(
        checkpoints,
        args.output,
        allow_unbound_checkpoints=args.allow_unbound_checkpoints,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "num_checkpoints": len(payload["averaged_checkpoints"]),
        "checkpoints": payload["averaged_checkpoints"],
        "split_sha256": payload["split_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
