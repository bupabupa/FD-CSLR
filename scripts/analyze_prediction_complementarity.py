import argparse
import json
from pathlib import Path

from evaluation.frequency_metrics import sequence_error_counts


def load_predictions(path):
    records = {}
    with open(path, encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise ValueError(
                    "duplicate sample_id {} in {}".format(sample_id, path)
                )
            records[sample_id] = record
    return records


def edit_distance(reference, hypothesis):
    num_classes = max(reference + hypothesis, default=0) + 1
    return int(sequence_error_counts(
        reference,
        hypothesis,
        num_classes,
    )[0])


def analyze(named_predictions):
    names = list(named_predictions)
    maps = {
        name: load_predictions(path)
        for name, path in named_predictions.items()
    }
    sample_sets = {name: set(records) for name, records in maps.items()}
    first_samples = sample_sets[names[0]]
    for name in names[1:]:
        if sample_sets[name] != first_samples:
            raise ValueError("prediction files do not contain identical samples")
    sample_ids = sorted(first_samples)

    per_model = {}
    per_model_edits = {}
    for name in names:
        edits = 0
        reference_length = 0
        for sample_id in sample_ids:
            record = maps[name][sample_id]
            reference = record["reference_ids"]
            hypothesis = record["hypothesis_ids"]
            sample_edits, sample_length = sequence_error_counts(
                reference,
                hypothesis,
                max(reference + hypothesis, default=0) + 1,
            )
            edits += int(sample_edits)
            reference_length += int(sample_length)
        per_model_edits[name] = edits
        per_model[name] = {
            "edits": edits,
            "reference_length": reference_length,
            "wer": (
                100.0 * edits / reference_length
                if reference_length
                else 0.0
            ),
        }

    oracle_edits = 0
    reference_length = 0
    winners = {name: 0 for name in names}
    baseline_wins = {name: 0 for name in names[1:]}
    for sample_id in sample_ids:
        reference = maps[names[0]][sample_id]["reference_ids"]
        reference_length += len(reference)
        distances = {
            name: edit_distance(
                reference,
                maps[name][sample_id]["hypothesis_ids"],
            )
            for name in names
        }
        best_edits = min(distances.values())
        oracle_edits += best_edits
        for name, edits in distances.items():
            if edits == best_edits:
                winners[name] += 1
        baseline = distances[names[0]]
        for name in names[1:]:
            if distances[name] < baseline:
                baseline_wins[name] += 1

    return {
        "samples": len(sample_ids),
        "models": per_model,
        "oracle": {
            "edits": oracle_edits,
            "reference_length": reference_length,
            "wer": (
                100.0 * oracle_edits / reference_length
                if reference_length
                else 0.0
            ),
            "absolute_gain_from_baseline": (
                per_model[names[0]]["wer"]
                - 100.0 * oracle_edits / reference_length
                if reference_length
                else 0.0
            ),
        },
        "ties_for_best": winners,
        "other_models_strictly_better_than_baseline_sentences": baseline_wins,
        "edits_vs_baseline": {
            name: per_model_edits[name] - per_model_edits[names[0]]
            for name in names[1:]
        },
    }


def parse_named_path(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    path = Path(path).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(
            "prediction file does not exist: {}".format(path)
        )
    return name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        type=parse_named_path,
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    named_predictions = dict(args.prediction)
    if len(named_predictions) != len(args.prediction):
        parser.error("prediction names must be unique")
    result = analyze(named_predictions)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
