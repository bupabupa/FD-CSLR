import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.frequency_metrics import sequence_error_counts


def read_predictions(path):
    samples = {}
    with open(path, encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in samples:
                raise ValueError(f"duplicate prediction id: {sample_id}")
            samples[sample_id] = row
    return samples


def sentence_statistics(samples, num_classes):
    sample_ids = sorted(samples)
    edits = np.zeros(len(sample_ids), dtype=np.int64)
    lengths = np.zeros(len(sample_ids), dtype=np.int64)
    for index, sample_id in enumerate(sample_ids):
        row = samples[sample_id]
        edits[index], lengths[index] = sequence_error_counts(
            row["reference_ids"],
            row["hypothesis_ids"],
            num_classes,
        )
    return sample_ids, edits, lengths


def grouped_statistics(samples, num_classes, field):
    groups = {}
    for sample_id, row in samples.items():
        group = str(row.get(field, "unknown") or "unknown")
        edits, reference_length = sequence_error_counts(
            row["reference_ids"],
            row["hypothesis_ids"],
            num_classes,
        )
        group_values = groups.setdefault(
            group, {"edits": 0, "reference_occurrences": 0, "samples": 0}
        )
        group_values["edits"] += edits
        group_values["reference_occurrences"] += reference_length
        group_values["samples"] += 1
    for values in groups.values():
        values["wer"] = float(
            100.0
            * values["edits"]
            / max(values["reference_occurrences"], 1)
        )
    return dict(sorted(groups.items()))


def bootstrap_wer(edits, lengths, iterations, seed):
    rng = np.random.default_rng(seed)
    values = np.zeros(iterations, dtype=np.float64)
    sample_count = len(edits)
    for index in range(iterations):
        sample = rng.integers(0, sample_count, size=sample_count)
        numerator = edits[sample].sum()
        denominator = max(lengths[sample].sum(), 1)
        values[index] = 100.0 * numerator / denominator
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
    }


def bootstrap_difference(
    reference_edits,
    reference_lengths,
    candidate_edits,
    candidate_lengths,
    iterations,
    seed,
):
    rng = np.random.default_rng(seed + 1)
    values = np.zeros(iterations, dtype=np.float64)
    sample_count = len(reference_edits)
    for index in range(iterations):
        sample = rng.integers(0, sample_count, size=sample_count)
        reference_wer = (
            reference_edits[sample].sum()
            / max(reference_lengths[sample].sum(), 1)
        )
        candidate_wer = (
            candidate_edits[sample].sum()
            / max(candidate_lengths[sample].sum(), 1)
        )
        values[index] = 100.0 * (candidate_wer - reference_wer)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
        "probability_improvement": float((values < 0.0).mean()),
        "p_value_one_sided": float(
            (1 + np.count_nonzero(values >= 0.0))
            / (len(values) + 1)
        ),
    }


def holm_adjust(p_values):
    values = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running = 0.0
    total = len(values)
    for rank, (name, p_value) in enumerate(values):
        running = max(running, min(1.0, (total - rank) * float(p_value)))
        adjusted[name] = running
    return adjusted


def _named_paths(values):
    output = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("input must use NAME=PATH")
        output[name] = Path(path)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction", action="append", required=True
    )
    parser.add_argument("--reference", default=None)
    parser.add_argument("--num-classes", type=int, default=1296)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = _named_paths(args.prediction)
    reference_name = args.reference or next(iter(paths))
    if reference_name not in paths:
        raise ValueError("reference name is absent from prediction inputs")
    results = {}
    statistics = {}
    for name, path in paths.items():
        samples = read_predictions(path)
        ids, edits, lengths = sentence_statistics(
            samples, args.num_classes
        )
        statistics[name] = (ids, edits, lengths)
        frequency_path = path.with_name(
            path.name.replace("predictions-", "frequency-metrics-")
            .replace(".jsonl", ".json")
        )
        frequency = None
        if frequency_path.exists():
            with open(frequency_path, encoding="utf-8") as source:
                frequency = json.load(source)
        results[name] = {
            "num_sentences": int(len(ids)),
            "edits": int(edits.sum()),
            "reference_occurrences": int(lengths.sum()),
            "wer": float(100.0 * edits.sum() / max(lengths.sum(), 1)),
            "bootstrap_wer": bootstrap_wer(
                edits, lengths, args.bootstrap, args.seed
            ),
            "signer_groups": grouped_statistics(
                samples, args.num_classes, "signer"
            ),
            "frequency_metrics": frequency,
        }

    reference_ids, reference_edits, reference_lengths = statistics[
        reference_name
    ]
    paired_results = {}
    for name, (ids, edits, lengths) in statistics.items():
        if name == reference_name:
            continue
        if ids != reference_ids:
            raise ValueError("paired predictions have different sample ids")
        paired_results[name] = bootstrap_difference(
            reference_edits,
            reference_lengths,
            edits,
            lengths,
            args.bootstrap,
            args.seed,
        )
    adjusted = holm_adjust({
        name: result["p_value_one_sided"]
        for name, result in paired_results.items()
    })
    for name, result in paired_results.items():
        result["p_value_holm"] = adjusted[name]
        results[name][
            "paired_wer_difference_vs_" + reference_name
        ] = result
    output = {
        "reference": reference_name,
        "bootstrap_iterations": args.bootstrap,
        "seed": args.seed,
        "models": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
