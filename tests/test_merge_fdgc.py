from pathlib import Path

import torch

from scripts.merge_fdgc_checkpoint import merge_checkpoints


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "preprocess" / "phoenix2014" / "split_manifest.json"


def test_merge_fdgc_checkpoint():
    from dataset.splits import load_split_manifest

    split_hash = load_split_manifest(MANIFEST)["split_sha256"]
    base = {
        "model_state_dict": {"encoder.weight": torch.ones(1)},
        "split_sha256": split_hash,
    }
    fdgc = {
        "fdgc_state_dict": {
            "class_mlp.0.weight": torch.zeros(2, 2),
        },
        "fdgc_difficulty": torch.linspace(0.0, 1.0, 4),
        "identity_metrics": {"ece": 0.1},
        "best_loss": 0.5,
        "split_sha256": split_hash,
    }
    merged = merge_checkpoints(base, fdgc, MANIFEST)
    assert "fdgc.class_mlp.0.weight" in merged["model_state_dict"]
    assert "fdgc_difficulty" in merged["model_state_dict"]
    assert merged["fdgc_metrics"]["best_loss"] == 0.5


if __name__ == "__main__":
    test_merge_fdgc_checkpoint()
    print("FDGC checkpoint merge test passed")
