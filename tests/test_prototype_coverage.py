import json
import tempfile
from pathlib import Path

import torch

from scripts.export_prototype_coverage import export_coverage


def test_export_prototype_coverage():
    checkpoint = {
        "split_sha256": "abc123",
        "model_state_dict": {
            "prototype_head.prototypes": torch.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
            ]),
            "prototype_head.support": torch.tensor([0.0, 0.1, 1.0]),
            "prototype_head.variances": torch.ones(3, 2),
            "prototype_gate": torch.tensor(0.0),
            "fdgc_difficulty": torch.tensor([0.0, 0.5]),
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(checkpoint, path)
        result = export_coverage(path, min_support=0.05)

    assert result["split_sha256"] == "abc123"
    assert result["num_classes"] == 3
    assert result["established_classes"] == 2
    assert result["classes"][0]["established"] is False
    assert result["classes"][2]["established"] is True
    assert result["prototype_gate_probability"] == 0.5
    assert result["fdgc_difficulty_nonzero_classes"] == 1


def test_export_coverage_writes_json():
    checkpoint = {
        "model_state_dict": {
            "prototype_head.prototypes": torch.ones(2, 3),
            "prototype_head.support": torch.ones(2),
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "checkpoint.pt"
        output_path = Path(directory) / "coverage.json"
        torch.save(checkpoint, checkpoint_path)
        result = export_coverage(checkpoint_path)
        output_path.write_text(json.dumps(result), encoding="utf-8")
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded["established_classes"] == 2


if __name__ == "__main__":
    test_export_prototype_coverage()
    test_export_coverage_writes_json()
    print("prototype coverage tests passed")
