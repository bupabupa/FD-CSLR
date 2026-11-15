from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.merge_fdgc_checkpoint import _load_checkpoint


def test_checkpoint_loader_accepts_numpy_rng_state():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": {
                    "weight": torch.ones(1),
                },
                "rng_state": {
                    "numpy": np.random.RandomState(0).get_state(),
                },
            },
            path,
        )
        payload = _load_checkpoint(path)
    assert payload["model_state_dict"]["weight"].item() == 1.0
    assert "numpy" in payload["rng_state"]


if __name__ == "__main__":
    test_checkpoint_loader_accepts_numpy_rng_state()
    print("checkpoint loader compatibility tests passed")
