import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import load_split_manifest


def _load_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint
    try:
        return torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch.load(checkpoint, map_location="cpu")


def merge_checkpoints(base_checkpoint, fdgc_checkpoint, split_manifest):
    base = _load_checkpoint(base_checkpoint)
    fdgc = _load_checkpoint(fdgc_checkpoint)
    manifest = load_split_manifest(split_manifest)
    for name, artifact in (("base", base), ("fdgc", fdgc)):
        actual = artifact.get("split_sha256")
        if actual != manifest["split_sha256"]:
            raise ValueError(
                f"{name} checkpoint split hash does not match manifest"
            )
    if "model_state_dict" not in base:
        raise ValueError("base checkpoint is missing model_state_dict")
    state_dict = base["model_state_dict"]
    for name, tensor in fdgc["fdgc_state_dict"].items():
        state_dict[f"fdgc.{name}"] = tensor
    state_dict["fdgc_difficulty"] = fdgc["fdgc_difficulty"]
    base["model_state_dict"] = state_dict
    base["fdgc_metrics"] = {
        "identity": fdgc.get("identity_metrics"),
        "best_loss": fdgc.get("best_loss"),
    }
    base["split_sha256"] = manifest["split_sha256"]
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--fdgc-checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = merge_checkpoints(
        args.base_checkpoint,
        args.fdgc_checkpoint,
        args.split_manifest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
