import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_signer_diagnostics.py"
sys.path.insert(0, str(ROOT))

from scripts.evaluate_signer_invariance import (
    evaluate_signer_swap,
    fit_signer_probe,
    masked_mean,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_signer_diagnostics():
    baseline = {
        "split_sha256": "same-split",
        "signer_probe": {
            "available": True,
            "accuracy": 0.5,
            "macro_recall": 0.45,
            "nll": 1.2,
        },
        "signer_swap": {"available": False},
    }
    h1 = {
        "split_sha256": "same-split",
        "signer_probe": {
            "available": True,
            "accuracy": 0.3,
            "macro_recall": 0.25,
            "nll": 1.5,
        },
        "signer_swap": {
            "available": True,
            "mean_kl": 0.01,
            "mean_js": 0.002,
            "mean_l1": 0.03,
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        baseline_path = directory / "baseline.json"
        h1_path = directory / "h1.json"
        output_path = directory / "comparison.json"
        write_json(baseline_path, baseline)
        write_json(h1_path, h1)
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--baseline",
                str(baseline_path),
                "--h1",
                str(h1_path),
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        comparison = json.loads(output_path.read_text(encoding="utf-8"))
    probe = comparison["metrics"]["accuracy"]
    assert probe["baseline"] == 0.5
    assert probe["h1"] == 0.3
    assert abs(probe["h1_minus_baseline"] + 0.2) < 1e-12
    assert comparison["metrics"]["swap_mean_js"]["baseline"] is None
    print("signer diagnostic comparison test passed", flush=True)


def test_probe_and_swap_helpers():
    features = torch.tensor(
        [
            [2.0, 0.0],
            [1.8, 0.2],
            [1.9, -0.1],
            [-2.0, 0.0],
            [-1.8, -0.2],
            [-1.9, 0.1],
        ]
    )
    signers = torch.tensor([1, 1, 1, 2, 2, 2])
    probe = fit_signer_probe(
        features[:4],
        signers[:4],
        features[4:],
        signers[4:],
        epochs=100,
        learning_rate=0.05,
        weight_decay=0.0,
    )
    assert probe["available"] is True
    assert probe["accuracy"] == 1.0

    sequence = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0]], [[7.0, 11.0], [13.0, 17.0]]]
    )
    pooled = masked_mean(sequence, torch.tensor([1, 2]))
    torch.testing.assert_close(
        pooled, torch.tensor([[1.0, 2.0], [8.0, 11.0]])
    )

    class DummySignerModel(torch.nn.Module):
        supports_signer = True
        signer_seen = torch.tensor([False, True, True])

        def forward(
            self, video, length, decode=False, signer_ids=None
        ):
            logits = torch.zeros(
                length.max().item(), signer_ids.numel(), 3
            )
            logits[:, :, 1] = torch.where(
                signer_ids == 1, 2.0, -2.0
            )
            logits[:, :, 2] = -logits[:, :, 1]
            return {"sequence_logits": logits, "feat_len": length}

    batch = (
        torch.zeros(2, 1, 1, 1, 1),
        torch.tensor([2, 2]),
        None,
        None,
        torch.tensor([1, 2]),
    )
    swap = evaluate_signer_swap(
        DummySignerModel(),
        [batch],
        torch.device("cpu"),
        {1: 2, 2: 1},
    )
    assert swap["available"] is True
    assert swap["mean_kl"] > 0.0
    assert swap["mean_js"] > 0.0
    print("signer probe and swap helper test passed", flush=True)


if __name__ == "__main__":
    test_compare_signer_diagnostics()
    test_probe_and_swap_helpers()
    print("signer diagnostics tests passed", flush=True)
