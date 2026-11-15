import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import (
    indexed_record_count,
    load_split_info,
    load_split_manifest,
)
from evaluation.frequency_metrics import gloss_occurrence_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess-dir",
        default=str(ROOT / "preprocess" / "phoenix2014"),
    )
    parser.add_argument(
        "--split-manifest",
        default=str(
            ROOT / "preprocess" / "phoenix2014" / "split_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "preprocess" / "phoenix2014" / "train_prior_split0.npy"
        ),
    )
    parser.add_argument("--smoothing", type=float, default=1.0)
    args = parser.parse_args()

    preprocess_dir = Path(args.preprocess_dir)
    manifest = load_split_manifest(args.split_manifest, preprocess_dir)
    gloss_dict = np.load(
        preprocess_dir / "gloss_dict.npy", allow_pickle=True
    ).item()
    train_info = load_split_info(
        preprocess_dir, "train", args.split_manifest
    )
    counts = gloss_occurrence_counts(train_info, gloss_dict)
    semantic_counts = counts[1:].astype(np.float64)
    if args.smoothing < 0:
        raise ValueError("smoothing cannot be negative")
    semantic_prior = semantic_counts + float(args.smoothing)
    semantic_prior /= semantic_prior.sum()
    prior = np.zeros(len(counts), dtype=np.float64)
    prior[1:] = semantic_prior
    np.save(args.output, prior.astype(np.float32))

    report = {
        "split_sha256": manifest["split_sha256"],
        "num_classes": int(len(prior)),
        "num_train_samples": int(indexed_record_count(train_info)),
        "num_gloss_occurrences": int(semantic_counts.sum()),
        "smoothing": float(args.smoothing),
        "num_missing_classes": int((semantic_counts == 0).sum()),
        "max_prior": float(semantic_prior.max()),
        "min_prior": float(semantic_prior.min()),
    }
    report_path = Path(args.output).with_suffix(".json")
    with open(report_path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    print(f"prior: {args.output}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
