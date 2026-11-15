from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.merge_multichannel_checkpoint import merge_checkpoints
from main import Processor


MANIFEST = ROOT / "preprocess" / "phoenix2014" / "split_manifest.json"


def test_merge_multichannel_checkpoint():
    from dataset.splits import load_split_manifest

    split_hash = load_split_manifest(MANIFEST)["split_sha256"]
    base = {
        "model_state_dict": {"encoder.weight": torch.ones(1)},
        "split_sha256": split_hash,
    }
    calibration = {
        "multichannel_state_dict": {
            "log_temperatures": torch.zeros(3),
            "fusion_logits": torch.ones(3),
            "fusion_parameterization_version": torch.tensor(2),
        },
        "best_loss": 0.5,
        "best_dev_wer": 20.0,
        "identity_dev_wer": 21.0,
        "selected_channel": "calibrated",
        "use_calibration": True,
        "split_sha256": split_hash,
    }
    merged = merge_checkpoints(base, calibration, MANIFEST)
    assert "multichannel_calibrator.log_temperatures" in merged[
        "model_state_dict"
    ]
    assert "multichannel_calibrator.fusion_logits" in merged[
        "model_state_dict"
    ]
    assert merged["multichannel_parameterization_version"] == 2
    assert merged["multichannel_metrics"]["best_loss"] == 0.5


def test_merge_rejects_unselected_calibration():
    from dataset.splits import load_split_manifest

    split_hash = load_split_manifest(MANIFEST)["split_sha256"]
    base = {
        "model_state_dict": {},
        "split_sha256": split_hash,
    }
    calibration = {
        "multichannel_state_dict": {},
        "use_calibration": False,
        "split_sha256": split_hash,
    }
    try:
        merge_checkpoints(base, calibration, MANIFEST)
    except ValueError:
        return
    raise AssertionError("unselected calibration was merged")


def test_merge_rejects_legacy_fusion_parameterization():
    from dataset.splits import load_split_manifest

    split_hash = load_split_manifest(MANIFEST)["split_sha256"]
    base = {
        "model_state_dict": {},
        "split_sha256": split_hash,
    }
    calibration = {
        "multichannel_state_dict": {
            "log_temperatures": torch.zeros(3),
            "fusion_logits": torch.ones(3),
        },
        "use_calibration": True,
        "split_sha256": split_hash,
    }
    try:
        merge_checkpoints(base, calibration, MANIFEST)
    except ValueError as error:
        assert "parameterization version" in str(error)
        return
    raise AssertionError("legacy fusion parameterization was merged")


def test_main_rejects_legacy_fusion_checkpoint():
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
        processor = Processor.__new__(Processor)
        processor.arg = SimpleNamespace(
            split_manifest=None,
            ignore_weights=[],
        )
        try:
            processor.load_model_weights(object(), checkpoint)
        except ValueError as error:
            assert "metadata version" in str(error)
            return
    raise AssertionError("main accepted a legacy fusion checkpoint")


if __name__ == "__main__":
    test_merge_multichannel_checkpoint()
    test_merge_rejects_unselected_calibration()
    test_merge_rejects_legacy_fusion_parameterization()
    test_main_rejects_legacy_fusion_checkpoint()
    print("multichannel checkpoint merge tests passed")
