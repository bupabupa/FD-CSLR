import copy
import json
import tempfile
from pathlib import Path

from dataset.splits import (
    build_leave_one_signer_manifest,
    build_phoenix_split_manifest,
    load_split_info,
    validate_split_manifest,
    write_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "preprocess" / "phoenix2014"


def test_phoenix_split_manifest():
    manifest = build_phoenix_split_manifest(
        PREPROCESS, calibration_fraction=0.1, seed=0
    )
    expected_sizes = {"train": 5104, "cal": 567, "sel": 540, "test": 629}
    assert {
        name: len(values) for name, values in manifest["splits"].items()
    } == expected_sizes
    validate_split_manifest(manifest, PREPROCESS)

    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "split_manifest.json"
        write_split_manifest(manifest, manifest_path)
        train_info = load_split_info(
            PREPROCESS, "train", manifest_path
        )
        calibration_info = load_split_info(
            PREPROCESS, "cal", manifest_path
        )
        assert len(train_info) == expected_sizes["train"] + 1
        assert len(calibration_info) == expected_sizes["cal"] + 1
        assert train_info["prefix"]
        assert calibration_info["prefix"]

        with open(manifest_path, encoding="utf-8") as source:
            tampered = json.load(source)
        tampered = copy.deepcopy(tampered)
        tampered["splits"]["cal"] = tampered["splits"]["cal"][1:]
        try:
            validate_split_manifest(tampered, PREPROCESS)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered split manifest was accepted")


def test_leave_one_signer_manifest():
    manifest = build_leave_one_signer_manifest(
        PREPROCESS,
        heldout_signer="Signer01",
        calibration_fraction=0.1,
        selection_fraction=0.4,
        seed=0,
    )
    assert manifest["heldout_signer"] == "Signer01"
    assert set(manifest["source_splits"].values()) == {"train"}
    validate_split_manifest(manifest, PREPROCESS)
    assert not set(manifest["splits"]["train"]).intersection(
        manifest["splits"]["sel"]
    )
    assert not set(manifest["splits"]["cal"]).intersection(
        manifest["splits"]["test"]
    )
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "leave_one_signer.json"
        write_split_manifest(manifest, manifest_path)
        for split_name in ("train", "cal", "sel", "test"):
            records = load_split_info(
                PREPROCESS, split_name, manifest_path
            )
            assert len(records) == len(manifest["splits"][split_name]) + 1
        selection = load_split_info(PREPROCESS, "sel", manifest_path)
        test = load_split_info(PREPROCESS, "test", manifest_path)
        training = load_split_info(PREPROCESS, "train", manifest_path)
        for records in (selection, test):
            assert {
                record["signer"]
                for index, record in records.items()
                if isinstance(index, int)
            } == {"Signer01"}
        assert "Signer01" not in {
            record["signer"]
            for index, record in training.items()
            if isinstance(index, int)
        }


if __name__ == "__main__":
    test_phoenix_split_manifest()
    test_leave_one_signer_manifest()
    print("split manifest tests passed")
