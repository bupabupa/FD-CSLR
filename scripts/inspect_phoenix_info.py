from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

for split in ("train", "dev", "test"):
    data = np.load(
        ROOT / "preprocess" / "phoenix2014" / f"{split}_info.npy",
        allow_pickle=True,
    ).item()
    first_key = next(key for key in data if isinstance(key, int))
    print(split, type(data).__name__, len(data), repr(first_key))
    print(data[first_key])
