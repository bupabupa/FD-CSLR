import hashlib
import json
from collections import Counter, OrderedDict
from numbers import Integral
from pathlib import Path

import numpy as np


PROTOCOL_VERSION = "fd_xmp_v3"
SOURCE_SPLITS = {
    "train": "train",
    "cal": "train",
    "sel": "dev",
    "test": "test",
}


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_file(preprocess_dir, filename):
    base = Path(preprocess_dir)
    for directory in (base, *base.parents):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "source split file not found from {}: {}".format(base, filename)
    )


def load_info(path):
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"PHOENIX info must be a dictionary: {path}")
    return data


def numeric_records(info):
    return {
        int(index): record
        for index, record in info.items()
        if isinstance(index, Integral) and not isinstance(index, bool)
    }


def indexed_record_count(inputs):
    """Count records by their indexed keys, ignoring metadata entries."""
    if not isinstance(inputs, dict):
        return len(inputs)
    keys = sorted(
        int(key)
        for key in inputs
        if isinstance(key, Integral) and not isinstance(key, bool)
    )
    if keys != list(range(len(keys))):
        raise ValueError(
            "indexed records must use contiguous non-negative integer keys"
        )
    return len(keys)


def sample_id(record):
    file_id = record.get("fileid")
    if not file_id:
        raise ValueError("PHOENIX sample is missing fileid")
    return str(file_id)


def deterministic_order(records, seed):
    def key(item):
        record_id, record = item
        value = f"{int(seed)}:{sample_id(record)}".encode("utf-8")
        return hashlib.sha256(value).hexdigest(), record_id

    return sorted(records.items(), key=key)


def _label_coverage(records):
    labels = Counter()
    for record in records.values():
        labels.update(record.get("label", "").split())
    return {
        "num_gloss_types": int(len(labels)),
        "num_gloss_occurrences": int(sum(labels.values())),
    }


def build_phoenix_split_manifest(
    preprocess_dir,
    calibration_fraction=0.1,
    seed=0,
):
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    preprocess_dir = Path(preprocess_dir)
    source_infos = {
        name: load_info(preprocess_dir / f"{name}_info.npy")
        for name in ("train", "dev", "test")
    }
    records = {
        "train": numeric_records(source_infos["train"]),
        "sel": numeric_records(source_infos["dev"]),
        "test": numeric_records(source_infos["test"]),
    }
    ordered_train = deterministic_order(records["train"], seed)
    calibration_size = max(
        1, int(round(len(ordered_train) * float(calibration_fraction)))
    )
    calibration_items = ordered_train[:calibration_size]
    model_train_items = ordered_train[calibration_size:]
    split_records = {
        "train": dict(model_train_items),
        "cal": dict(calibration_items),
        "sel": records["sel"],
        "test": records["test"],
    }
    splits = {
        name: sorted(sample_id(record) for record in values.values())
        for name, values in split_records.items()
    }
    all_ids = [item for values in splits.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("PHOENIX split sources overlap by fileid")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "splits": splits,
    }
    split_digest = _sha256_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return {
        **payload,
        "calibration_fraction": float(calibration_fraction),
        "source_splits": SOURCE_SPLITS,
        "source_files": {
            f"{name}_info.npy": file_sha256(
                preprocess_dir / f"{name}_info.npy"
            )
            for name in ("train", "dev", "test")
        },
        "statistics": {
            name: {
                "num_samples": len(values),
                "signers": dict(sorted(Counter(
                    record.get("signer", "unknown")
                    for record in split_records[name].values()
                ).items())),
                **_label_coverage(split_records[name]),
            }
            for name, values in split_records.items()
        },
        "split_sha256": split_digest,
    }


def build_leave_one_signer_manifest(
    preprocess_dir,
    heldout_signer,
    calibration_fraction=0.1,
    selection_fraction=0.4,
    seed=0,
    minimum_heldout_samples=50,
):
    """Build a deterministic signer-heldout protocol from the train source."""
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    if not 0.0 < float(selection_fraction) < 1.0:
        raise ValueError("selection_fraction must be between zero and one")
    preprocess_dir = Path(preprocess_dir)
    train_records = numeric_records(
        load_info(preprocess_dir / "train_info.npy")
    )
    heldout_records = {
        index: record
        for index, record in train_records.items()
        if str(record.get("signer")) == str(heldout_signer)
    }
    remaining_records = {
        index: record
        for index, record in train_records.items()
        if str(record.get("signer")) != str(heldout_signer)
    }
    if len(heldout_records) < int(minimum_heldout_samples):
        raise ValueError(
            "held-out signer has {} samples, below minimum {}".format(
                len(heldout_records), int(minimum_heldout_samples)
            )
        )
    if not remaining_records:
        raise ValueError("no training samples remain after signer holdout")

    ordered_remaining = deterministic_order(remaining_records, seed)
    calibration_size = max(
        1,
        int(round(len(ordered_remaining) * float(calibration_fraction))),
    )
    if calibration_size >= len(ordered_remaining):
        raise ValueError("calibration fraction leaves no model-training data")
    calibration_items = ordered_remaining[:calibration_size]
    model_train_items = ordered_remaining[calibration_size:]

    ordered_heldout = deterministic_order(heldout_records, seed)
    selection_size = max(
        1,
        int(round(len(ordered_heldout) * float(selection_fraction))),
    )
    if selection_size >= len(ordered_heldout):
        raise ValueError("selection fraction leaves no held-out Test data")
    selection_items = ordered_heldout[:selection_size]
    heldout_test_items = ordered_heldout[selection_size:]
    split_records = {
        "train": dict(model_train_items),
        "cal": dict(calibration_items),
        "sel": dict(selection_items),
        "test": dict(heldout_test_items),
    }
    splits = {
        name: sorted(sample_id(record) for record in values.values())
        for name, values in split_records.items()
    }
    all_ids = [item for values in splits.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("leave-one-signer split sources overlap by fileid")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "splits": splits,
    }
    split_digest = _sha256_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return {
        **payload,
        "calibration_fraction": float(calibration_fraction),
        "selection_fraction": float(selection_fraction),
        "heldout_signer": str(heldout_signer),
        "source_splits": {
            "train": "train",
            "cal": "train",
            "sel": "train",
            "test": "train",
        },
        "source_files": {
            "train_info.npy": file_sha256(
                preprocess_dir / "train_info.npy"
            )
        },
        "statistics": {
            name: {
                "num_samples": len(values),
                "signers": dict(sorted(Counter(
                    record.get("signer", "unknown")
                    for record in split_records[name].values()
                ).items())),
                **_label_coverage(split_records[name]),
            }
            for name, values in split_records.items()
        },
        "split_sha256": split_digest,
    }


def write_split_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return path


def validate_split_manifest(manifest, preprocess_dir=None):
    required = {
        "protocol_version",
        "seed",
        "splits",
        "split_sha256",
        "source_splits",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(
            f"split manifest is missing fields: {sorted(missing)}"
        )
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unexpected split protocol version")
    expected_names = set(SOURCE_SPLITS)
    if set(manifest["splits"]) != expected_names:
        raise ValueError("split manifest has unexpected split names")
    if set(manifest["source_splits"]) != expected_names:
        raise ValueError("split manifest has unexpected source mappings")
    allowed_sources = set(SOURCE_SPLITS.values())
    for name, source_name in manifest["source_splits"].items():
        if source_name not in allowed_sources:
            raise ValueError(
                "unexpected source split for {}: {}".format(
                    name, source_name
                )
            )
    all_ids = []
    for name, values in manifest["splits"].items():
        if values != sorted(values):
            raise ValueError(f"split {name} is not sorted")
        if len(values) != len(set(values)):
            raise ValueError(f"split {name} contains duplicate ids")
        all_ids.extend(values)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split manifest contains cross-split leakage")
    payload = {
        "protocol_version": manifest["protocol_version"],
        "seed": int(manifest["seed"]),
        "splits": manifest["splits"],
    }
    actual_digest = _sha256_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    if actual_digest != manifest["split_sha256"]:
        raise ValueError("split manifest hash mismatch")
    if preprocess_dir is not None and "source_files" in manifest:
        preprocess_dir = Path(preprocess_dir)
        for filename, expected in manifest["source_files"].items():
            actual = file_sha256(
                resolve_source_file(preprocess_dir, filename)
            )
            if actual != expected:
                raise ValueError(f"source split file changed: {filename}")
    return manifest


def load_split_manifest(path, preprocess_dir=None):
    with open(path, encoding="utf-8") as source:
        manifest = json.load(source)
    return validate_split_manifest(manifest, preprocess_dir)


def split_info_path(preprocess_dir, split_name, source_name=None):
    if split_name not in SOURCE_SPLITS:
        raise ValueError(f"unknown split name: {split_name}")
    if source_name is None:
        source_name = SOURCE_SPLITS[split_name]
    if source_name not in set(SOURCE_SPLITS.values()):
        raise ValueError(f"unknown source split name: {source_name}")
    return resolve_source_file(
        preprocess_dir, f"{source_name}_info.npy"
    )


def load_split_info(preprocess_dir, split_name, manifest_path):
    manifest = load_split_manifest(manifest_path, preprocess_dir)
    source_name = manifest["source_splits"][split_name]
    source_info = load_info(
        split_info_path(preprocess_dir, split_name, source_name)
    )
    records_by_id = {
        sample_id(record): record
        for record in numeric_records(source_info).values()
    }
    output = OrderedDict()
    if "prefix" in source_info:
        output["prefix"] = source_info["prefix"]
    for index, record_id in enumerate(manifest["splits"][split_name]):
        if record_id not in records_by_id:
            raise ValueError(
                f"{record_id} is absent from source split {source_name}"
            )
        output[index] = records_by_id[record_id]
    return output
