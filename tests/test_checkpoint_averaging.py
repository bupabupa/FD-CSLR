import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.average_checkpoints import (
    average_checkpoints,
    average_state_dicts,
    discover_checkpoints,
)
from utils.random_state import RandomState


def _save(path, value, split_sha256="split"):
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": {
                "weight": torch.tensor([value], dtype=torch.float32),
                "count": torch.tensor([7], dtype=torch.int64),
                "flag": torch.tensor([True], dtype=torch.bool),
            },
            "split_sha256": split_sha256,
            "rng_state": RandomState(seed=1).save_rng_state(),
            "optimizer_state_dict": {
                "large_padding": [1] * 1024,
            },
        },
        path,
    )


def test_discovery_and_average():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = [
            root / "dev_3.00_epoch1_model.pt",
            root / "dev_2.00_epoch2_model.pt",
            root / "dev_1.00_epoch3_model.pt",
        ]
        for index, path in enumerate(paths, start=1):
            _save(path, float(index))
        selected = discover_checkpoints(root, last_n=2)
        assert selected == paths[1:]
        output = root / "averaged.pt"
        payload = average_checkpoints(selected, output)
        assert set(payload) == {
            "epoch",
            "model_state_dict",
            "split_sha256",
            "diagnostic_unbound_checkpoints",
            "averaged_checkpoints",
            "average_rule",
        }
        assert payload["diagnostic_unbound_checkpoints"] is False
        assert torch.allclose(
            payload["model_state_dict"]["weight"],
            torch.tensor([2.5]),
        )
        assert torch.equal(
            payload["model_state_dict"]["count"],
            torch.tensor([7], dtype=torch.int64),
        )
        assert torch.equal(
            payload["model_state_dict"]["flag"],
            torch.tensor([True], dtype=torch.bool),
        )


def test_split_and_non_tensor_guards():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "dev_2.00_epoch1_model.pt"
        second = root / "dev_1.00_epoch2_model.pt"
        _save(first, 1.0, split_sha256="a")
        _save(second, 2.0, split_sha256="b")
        try:
            average_checkpoints([first, second], root / "out.pt")
        except ValueError as error:
            assert "split hashes" in str(error)
        else:
            raise AssertionError("split hash mismatch was not rejected")

        try:
            average_state_dicts([
                {"x": torch.zeros(1)},
                {"x": torch.ones(2)},
            ])
        except ValueError as error:
            assert "differs" in str(error)
        else:
            raise AssertionError("shape mismatch was not rejected")


def test_missing_split_hash_is_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "dev_1.00_epoch1_model.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": {"weight": torch.zeros(1)},
            },
            checkpoint,
        )
        try:
            average_checkpoints([checkpoint], root / "out.pt")
        except ValueError as error:
            assert "missing split_sha256" in str(error)
        else:
            raise AssertionError("missing split hash was not rejected")

        diagnostic = average_checkpoints(
            [checkpoint],
            root / "diagnostic.pt",
            allow_unbound_checkpoints=True,
        )
        assert diagnostic["split_sha256"] is None
        assert diagnostic["diagnostic_unbound_checkpoints"] is True


def test_legacy_checkpoint_requires_diagnostic_flag():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "dev_1.00_epoch1_model.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": {"weight": torch.zeros(1)},
                "legacy_rng_state": np.array([1, 2, 3]),
            },
            checkpoint,
        )
        try:
            average_checkpoints([checkpoint], root / "strict.pt")
        except (pickle.UnpicklingError, RuntimeError):
            pass
        else:
            raise AssertionError("legacy checkpoint loaded without diagnostic flag")

        payload = average_checkpoints(
            [checkpoint],
            root / "diagnostic.pt",
            allow_unbound_checkpoints=True,
        )
        assert payload["diagnostic_unbound_checkpoints"] is True


if __name__ == "__main__":
    test_discovery_and_average()
    test_split_and_non_tensor_guards()
    test_missing_split_hash_is_rejected()
    test_legacy_checkpoint_requires_diagnostic_flag()
    print("checkpoint averaging tests passed")
