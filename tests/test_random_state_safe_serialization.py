import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from utils.random_state import RandomState


def _next_values():
    return (
        np.random.random(8),
        torch.rand(8),
        [random.random() for _ in range(8)],
    )


def _assert_same(left, right):
    np.testing.assert_allclose(left[0], right[0])
    assert torch.equal(left[1], right[1])
    assert left[2] == right[2]


def test_rng_state_round_trip_and_weights_only_load():
    state = RandomState(seed=17)
    np.random.random(11)
    torch.rand(7)
    random.random()
    saved = state.save_rng_state()
    expected = _next_values()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save({
            "model_state_dict": {"weight": torch.zeros(1)},
            "rng_state": saved,
        }, checkpoint)
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

    restored = RandomState(seed=99)
    restored.set_rng_state(payload["rng_state"])
    _assert_same(expected, _next_values())


def test_legacy_numpy_rng_state_is_still_supported():
    state = RandomState(seed=23)
    np.random.random(5)
    torch.rand(3)
    random.random()
    legacy = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    expected = _next_values()
    state.set_rng_state(legacy)
    _assert_same(expected, _next_values())


if __name__ == "__main__":
    test_rng_state_round_trip_and_weights_only_load()
    test_legacy_numpy_rng_state_is_still_supported()
    print("safe RNG state serialization tests passed")
