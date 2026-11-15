#!/usr/bin/env python3
"""Choose between a single checkpoint and a parity-averaged checkpoint on Dev."""

import argparse
import json
from pathlib import Path
import re
import sys

import torch


DEV_WER_PATTERN = re.compile(
    r"\bdev\s+([0-9]+(?:\.[0-9]+)?)%",
    flags=re.IGNORECASE,
)


def read_dev_wer(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Dev metric file does not exist: {}".format(path))
    for line in reversed(
        path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        match = DEV_WER_PATTERN.search(line)
        if match:
            return float(match.group(1))
    raise ValueError("no authoritative Dev WER found in {}".format(path))


def checkpoint_split_hash(path):
    payload = torch.load(
        str(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    split_hash = payload.get("split_sha256")
    if not split_hash:
        raise ValueError("checkpoint is missing split_sha256: {}".format(path))
    return split_hash


def select_checkpoint(
    single_checkpoint,
    single_dev,
    average_checkpoint,
    average_dev,
):
    single_checkpoint = Path(single_checkpoint).resolve()
    average_checkpoint = Path(average_checkpoint).resolve()
    single_dev = Path(single_dev).resolve()
    average_dev = Path(average_dev).resolve()
    single_hash = checkpoint_split_hash(single_checkpoint)
    average_hash = checkpoint_split_hash(average_checkpoint)
    if single_hash != average_hash:
        raise ValueError(
            "single and averaged checkpoints use different split hashes"
        )
    single_wer = read_dev_wer(single_dev)
    average_wer = read_dev_wer(average_dev)
    use_average = average_wer < single_wer - 1e-9
    selected = average_checkpoint if use_average else single_checkpoint
    return {
        "selection_rule": (
            "use the arithmetic mean of the last N checkpoints only when its "
            "D_sel WER is strictly lower; exact ties fall back to the single "
            "checkpoint; D_test is forbidden"
        ),
        "split_sha256": single_hash,
        "single": {
            "checkpoint": str(single_checkpoint),
            "dev_metrics": str(single_dev),
            "dev_wer": single_wer,
        },
        "average": {
            "checkpoint": str(average_checkpoint),
            "dev_metrics": str(average_dev),
            "dev_wer": average_wer,
        },
        "selected": "average" if use_average else "single",
        "selected_checkpoint": str(selected),
        "dev_wer": average_wer if use_average else single_wer,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-checkpoint", required=True)
    parser.add_argument("--single-dev", required=True)
    parser.add_argument("--average-checkpoint", required=True)
    parser.add_argument("--average-dev", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = select_checkpoint(
        args.single_checkpoint,
        args.single_dev,
        args.average_checkpoint,
        args.average_dev,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result["selected_checkpoint"])


if __name__ == "__main__":
    main()
