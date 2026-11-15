import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_SELECTED_FIELDS = {
    "name",
    "dev_prediction",
    "config",
    "checkpoint",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_claim(lock_path):
    lock_path = Path(lock_path).expanduser().resolve()
    with open(lock_path, encoding="utf-8") as source:
        lock = json.load(source)
    if lock.get("protocol") != "A":
        raise ValueError("candidate lock is not a protocol A lock")
    if not lock.get("split_sha256"):
        raise ValueError("candidate lock is missing the split hash")
    selected = lock.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("candidate lock is missing selected")
    missing = REQUIRED_SELECTED_FIELDS.difference(selected)
    if missing:
        raise ValueError(
            "selected candidate is missing fields: {}".format(
                sorted(missing)
            )
        )
    for field in ("dev_prediction", "config", "checkpoint"):
        path = Path(selected[field]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                "selected candidate {} does not exist: {}".format(
                    field,
                    path,
                )
            )
        selected[field] = str(path)

    protocol_root = lock_path.parent.parent
    test_predictions = sorted(
        str(path)
        for path in protocol_root.rglob("predictions-test.jsonl")
        if path.is_file()
    )
    if test_predictions:
        raise ValueError(
            "Test predictions already exist before access claim: {}".format(
                ", ".join(test_predictions)
            )
        )
    return {
        "protocol": "A",
        "split_sha256": str(lock["split_sha256"]),
        "locked_candidate": str(lock_path),
        "locked_candidate_sha256": file_sha256(lock_path),
        "test_policy": "single_access_after_candidate_lock",
        "selected": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write the Test access claim; without this flag only verify",
    )
    args = parser.parse_args()
    claim = build_claim(args.lock)
    output = Path(args.output).expanduser().resolve()
    if args.execute:
        if output.exists():
            raise ValueError(
                "Test access claim already exists: {}".format(output)
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as destination:
            json.dump(claim, destination, indent=2, sort_keys=True)
            destination.write("\n")
    print(json.dumps(claim, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
