import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_protocol_manifest import validate_manifest


MANIFEST = ROOT / "protocols" / "a_protocol_manifest.json"


def test_protocol_a_manifest_is_ready():
    report = validate_manifest(MANIFEST, root=ROOT, require_runtime=True)
    assert report["protocol"] == "A"
    assert report["num_jobs"] == 9
    assert report["seeds"] == [0, 1, 2]
    assert set(report["roles"]) == {"baseline", "vac_ctc", "vac_smkd"}
    assert report["test_artifacts"] == []
    assert report["runtime"]["ctcdecode_version"] == "1.0.3"
    assert report["runtime"]["sclite_executable"]


def test_protocol_manifest_rejects_duplicate_job_names(tmp_path):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["jobs"][1]["name"] = payload["jobs"][0]["name"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_manifest(invalid, root=ROOT)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate job names were accepted")


if __name__ == "__main__":
    test_protocol_a_manifest_is_ready()
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_protocol_manifest_rejects_duplicate_job_names(
            __import__("pathlib").Path(directory)
        )
    print("protocol manifest tests passed")
