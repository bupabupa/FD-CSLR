import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import load_split_info, load_split_manifest


def _load_sequences(train_info, gloss_to_id):
    sequences = []
    unknown = set()
    tokens = 0
    for key, value in train_info.items():
        if key == "prefix":
            continue
        sequence = []
        for gloss in str(value["label"]).split():
            class_id = gloss_to_id.get(gloss)
            if class_id is None:
                unknown.add(gloss)
                continue
            sequence.append(int(class_id))
        if sequence:
            sequences.append(sequence)
            tokens += len(sequence)
    if unknown:
        raise ValueError(
            "training labels contain unknown glosses: {}".format(
                sorted(unknown)[:20]
            )
        )
    return sequences, tokens


def build_rescorer(
    gloss_dict_path,
    train_info,
    order,
    split_sha256=None,
):
    gloss_dict = np.load(
        gloss_dict_path, allow_pickle=True
    ).item()
    gloss_to_id = {
        str(gloss): int(value[0])
        for gloss, value in gloss_dict.items()
    }
    sequences, token_count = _load_sequences(train_info, gloss_to_id)
    return {
        "version": 1,
        "source": "D_train",
        "order": int(order),
        "blank_id": 0,
        "num_classes": int(len(gloss_dict) + 1),
        "split_sha256": split_sha256,
        "num_sequences": len(sequences),
        "num_gloss_tokens": token_count,
        "sequences": sequences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gloss-dict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--split-manifest")
    parser.add_argument("--train-info")
    args = parser.parse_args()
    if args.order < 1:
        raise ValueError("order must be positive")
    if bool(args.split_manifest) == bool(args.train_info):
        raise ValueError(
            "provide exactly one of --split-manifest and --train-info"
        )

    split_sha256 = None
    if args.split_manifest:
        manifest = load_split_manifest(args.split_manifest)
        train_info = load_split_info(
            str(Path(args.split_manifest).parent),
            "train",
            args.split_manifest,
        )
        split_sha256 = manifest["split_sha256"]
    else:
        train_info = np.load(
            args.train_info, allow_pickle=True
        ).item()

    payload = build_rescorer(
        args.gloss_dict,
        train_info,
        args.order,
        split_sha256=split_sha256,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(payload, destination, separators=(",", ":"))
        destination.write("\n")
    temporary.replace(output)
    print(
        "saved {} sequences and {} gloss tokens to {}".format(
            payload["num_sequences"],
            payload["num_gloss_tokens"],
            output,
        )
    )


if __name__ == "__main__":
    main()
