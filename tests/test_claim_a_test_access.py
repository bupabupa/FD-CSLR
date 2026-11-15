import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.claim_a_test_access import build_claim


def _write_lock(root):
    protocol_root = root / "A_protocol"
    lock_dir = protocol_root / "candidate_lock"
    lock_dir.mkdir(parents=True)
    prediction = protocol_root / "selection" / "predictions-dev.jsonl"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("{}\n", encoding="utf-8")
    config = root / "config.yaml"
    config.write_text("protocol: A\n", encoding="utf-8")
    checkpoint = root / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    lock_path = lock_dir / "locked_candidate.json"
    lock_path.write_text(
        json.dumps({
            "protocol": "A",
            "split_sha256": "locked-split",
            "selected": {
                "name": "candidate",
                "dev_prediction": str(prediction),
                "config": str(config),
                "checkpoint": str(checkpoint),
            },
        }),
        encoding="utf-8",
    )
    return protocol_root, lock_path


def test_claim_is_bound_to_existing_lock_and_has_no_test_prediction():
    with tempfile.TemporaryDirectory() as directory:
        protocol_root, lock_path = _write_lock(Path(directory))
        claim = build_claim(lock_path)
    assert claim["protocol"] == "A"
    assert claim["locked_candidate"] == str(lock_path.resolve())
    assert claim["selected"]["name"] == "candidate"
    assert protocol_root.name == "A_protocol"


def test_claim_rejects_preexisting_test_prediction():
    with tempfile.TemporaryDirectory() as directory:
        protocol_root, lock_path = _write_lock(Path(directory))
        leaked = protocol_root / "selection" / "predictions-test.jsonl"
        leaked.write_text("{}\n", encoding="utf-8")
        try:
            build_claim(lock_path)
        except ValueError as error:
            assert "Test predictions already exist" in str(error)
            return
    raise AssertionError("claim accepted a pre-existing Test prediction")


if __name__ == "__main__":
    test_claim_is_bound_to_existing_lock_and_has_no_test_prediction()
    test_claim_rejects_preexisting_test_prediction()
    print("A Test access claim tests passed")
