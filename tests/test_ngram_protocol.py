import json
from pathlib import Path
import tempfile

import yaml

from scripts.run_ngram_protocol import (
    make_rescored_config,
    register_optional_candidate,
)
from scripts.run_ngram_for_checkpoint import resolve_checkpoint


def test_rescored_config_preserves_model_and_sets_decoder():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / "base.yaml"
        base.write_text(
            yaml.safe_dump({
                "split_manifest": "manifest.json",
                "model_args": {
                    "num_classes": 1296,
                    "fdgc_enabled": True,
                },
            }),
            encoding="utf-8",
        )
        output = make_rescored_config(
            base,
            root / "out.yaml",
            root / "rescorer.json",
            alpha=0.002,
            beta=0.0,
        )
        result = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert result["model_args"]["num_classes"] == 1296
    assert result["model_args"]["fdgc_enabled"] is True
    assert result["model_args"]["beam_width"] == 50
    assert result["model_args"]["rescorer_alpha"] == 0.002


def test_optional_candidate_registration_replaces_same_name():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "optional_candidates.json"
        path.write_text(
            json.dumps([
                {"name": "other", "priority": 1},
                {"name": "candidate", "priority": 2},
            ]),
            encoding="utf-8",
        )
        register_optional_candidate(
            path,
            {"name": "candidate", "priority": 3},
        )
        result = json.loads(path.read_text(encoding="utf-8"))
    assert result == [
        {"name": "candidate", "priority": 3},
        {"name": "other", "priority": 1},
    ]


def test_checkpoint_resolution_prefers_explicit_then_calibrated():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / "run"
        base.mkdir()
        best = base / "dev_30.00_epoch1_model.pt"
        best.write_bytes(b"best")
        assert resolve_checkpoint(base, None) == best.resolve()

        calibrated = base / "fdgc" / "checkpoint_calibrated.pt"
        calibrated.parent.mkdir()
        calibrated.write_bytes(b"calibrated")
        assert resolve_checkpoint(base, None) == calibrated.resolve()

        explicit = root / "explicit.pt"
        explicit.write_bytes(b"explicit")
        assert resolve_checkpoint(base, explicit) == explicit.resolve()


if __name__ == "__main__":
    test_rescored_config_preserves_model_and_sets_decoder()
    test_optional_candidate_registration_replaces_same_name()
    test_checkpoint_resolution_prefers_explicit_then_calibrated()
    print("n-gram protocol tests passed")
