import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import load_split_manifest
from modules.calibration import (
    FUSION_PARAMETERIZATION_VERSION,
    validate_fusion_state,
)
from scripts.merge_fdgc_checkpoint import _load_checkpoint


def merge_checkpoints(base_checkpoint, calibration_checkpoint, split_manifest):
    base = _load_checkpoint(base_checkpoint)
    calibration = _load_checkpoint(calibration_checkpoint)
    manifest = load_split_manifest(split_manifest)
    for name, artifact in (
        ("base", base),
        ("calibration", calibration),
    ):
        if artifact.get("split_sha256") != manifest["split_sha256"]:
            raise ValueError(
                f"{name} checkpoint split hash does not match manifest"
            )
    if "model_state_dict" not in base:
        raise ValueError("base checkpoint is missing model_state_dict")
    if not calibration.get("use_calibration", False):
        raise ValueError(
            "calibration did not beat the base Dev candidate; "
            "refusing to enable it"
        )
    calibration_state = calibration.get("multichannel_state_dict")
    if not isinstance(calibration_state, dict):
        raise ValueError(
            "calibration checkpoint is missing multichannel_state_dict"
        )
    validate_fusion_state(calibration_state)
    state_dict = base["model_state_dict"]
    for name, tensor in calibration_state.items():
        state_dict[f"multichannel_calibrator.{name}"] = tensor
    base["model_state_dict"] = state_dict
    base["multichannel_parameterization_version"] = (
        FUSION_PARAMETERIZATION_VERSION
    )
    base["multichannel_metrics"] = {
        "best_loss": calibration.get("best_loss"),
        "best_dev_wer": calibration.get("best_dev_wer"),
        "identity_dev_wer": calibration.get("identity_dev_wer"),
        "selected_channel": calibration.get("selected_channel"),
    }
    base["split_sha256"] = manifest["split_sha256"]
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--calibration-checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = merge_checkpoints(
        args.base_checkpoint,
        args.calibration_checkpoint,
        args.split_manifest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
