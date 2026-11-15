import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.splits import load_split_info
from evaluation.frequency_metrics import (
    class_frequency_report,
    gloss_occurrence_counts,
)
from scripts.tune_ngram_rescorer import (
    make_sclite_evaluator,
    parse_float_list,
    write_predictions,
)


def edit_distance(left, right):
    left = list(left)
    right = list(right)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, start=1):
        current = [row]
        for column, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def normalized_distances(sequences, denominator):
    size = len(sequences)
    distances = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            distance = edit_distance(
                sequences[left],
                sequences[right],
            )
            if denominator == "max":
                length = max(
                    len(sequences[left]),
                    len(sequences[right]),
                    1,
                )
                value = distance / length
            elif denominator == "mean":
                length = max(
                    (len(sequences[left]) + len(sequences[right])) / 2.0,
                    1.0,
                )
                value = distance / length
            else:
                raise ValueError("unknown MBR denominator: {}".format(
                    denominator
                ))
            distances[left, right] = value
            distances[right, left] = value
    return distances


def mbr_select(
    beams,
    temperature=1.0,
    ctc_weight=0.05,
    denominator="max",
    beam_limit=50,
    distances=None,
):
    if not beams:
        raise ValueError("sample has no CTC beam candidates")
    candidates = beams[:int(beam_limit)]
    sequences = [list(beam["tokens"]) for beam in candidates]
    scores = np.asarray(
        [float(beam["ctc_score"]) for beam in candidates],
        dtype=np.float64,
    )
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    posterior = np.exp(
        -(scores - scores.min()) / temperature
    )
    posterior /= posterior.sum()
    if distances is None:
        distances = normalized_distances(sequences, denominator)
    else:
        distances = np.asarray(distances, dtype=np.float64)
        expected = (len(candidates), len(candidates))
        if distances.shape != expected:
            raise ValueError(
                "distance matrix shape {} does not match {}".format(
                    distances.shape,
                    expected,
                )
            )
    risk = distances.dot(posterior)
    relative_ctc_cost = scores - scores.min()
    combined = risk + float(ctc_weight) * relative_ctc_cost
    return int(np.argmin(combined))


def evaluate_selection(
    records,
    temperature,
    ctc_weight,
    denominator,
    train_counts,
    num_classes,
    distance_matrices=None,
):
    if len(train_counts) != int(num_classes):
        raise ValueError("train counts do not match num_classes")
    references = [
        list(map(int, record["reference_ids"]))
        for record in records
    ]
    selected_indices = [
        mbr_select(
            record["beams"],
            temperature=temperature,
            ctc_weight=ctc_weight,
            denominator=denominator,
            distances=(
                distance_matrices[index]
                if distance_matrices is not None
                else None
            ),
        )
        for index, record in enumerate(records)
    ]
    hypotheses = [
        list(record["beams"][index]["tokens"])
        for record, index in zip(records, selected_indices)
    ]
    report = class_frequency_report(
        references,
        hypotheses,
        train_counts,
        blank_id=0,
    )
    reference_occurrences = int(report["overall"]["support"])
    edits = (
        reference_occurrences
        - int(report["overall"]["true_positive"])
    )
    return {
        "temperature": float(temperature),
        "ctc_weight": float(ctc_weight),
        "denominator": denominator,
        "num_sentences": len(records),
        "edits": edits,
        "reference_occurrences": reference_occurrences,
        "wer": float(
            100.0 * edits / max(reference_occurrences, 1)
        ),
        "overall": report["overall"],
        "bins": report["bins"],
        "selected_ctc_beam_counts": {
            str(index): int(selected_indices.count(index))
            for index in sorted(set(selected_indices))
        },
        "hypotheses": hypotheses,
    }


def evaluate_ctc_baseline(records, train_counts, num_classes):
    if len(train_counts) != int(num_classes):
        raise ValueError("train counts do not match num_classes")
    references = [
        list(map(int, record["reference_ids"]))
        for record in records
    ]
    selected_indices = []
    for record in records:
        beams = record["beams"]
        if not beams:
            raise ValueError("sample has no CTC beam candidates")
        scores = np.asarray(
            [float(beam["ctc_score"]) for beam in beams],
            dtype=np.float64,
        )
        selected_indices.append(int(np.argmin(scores)))
    hypotheses = [
        list(record["beams"][index]["tokens"])
        for record, index in zip(records, selected_indices)
    ]
    report = class_frequency_report(
        references,
        hypotheses,
        train_counts,
        blank_id=0,
    )
    reference_occurrences = int(report["overall"]["support"])
    edits = (
        reference_occurrences
        - int(report["overall"]["true_positive"])
    )
    return {
        "temperature": 0.0,
        "ctc_weight": 1.0,
        "denominator": "none",
        "num_sentences": len(records),
        "edits": edits,
        "reference_occurrences": reference_occurrences,
        "wer": float(
            100.0 * edits / max(reference_occurrences, 1)
        ),
        "overall": report["overall"],
        "bins": report["bins"],
        "selected_ctc_beam_counts": {
            str(index): int(selected_indices.count(index))
            for index in sorted(set(selected_indices))
        },
        "hypotheses": hypotheses,
    }


def tune_grid(
    records,
    temperatures,
    ctc_weights,
    denominators,
    train_counts,
    num_classes,
    wer_evaluator,
):
    distance_by_denominator = {
        denominator: [
            normalized_distances(
                [
                    list(beam["tokens"])
                    for beam in record["beams"][:50]
                ],
                denominator,
            )
            for record in records
        ]
        for denominator in denominators
    }
    baseline = evaluate_ctc_baseline(
        records,
        train_counts=train_counts,
        num_classes=num_classes,
    )
    baseline["python_edit_wer"] = baseline["wer"]
    baseline["wer"] = float(wer_evaluator(baseline["hypotheses"]))
    baseline["wer_source"] = "sclite"
    candidates = []
    for temperature in temperatures:
        for ctc_weight in ctc_weights:
            for denominator in denominators:
                result = evaluate_selection(
                    records,
                    temperature,
                    ctc_weight,
                    denominator,
                    train_counts,
                    num_classes,
                    distance_matrices=distance_by_denominator[denominator],
                )
                result["python_edit_wer"] = result["wer"]
                result["wer"] = float(
                    wer_evaluator(result["hypotheses"])
                )
                result["wer_source"] = "sclite"
                result["eligible"] = bool(
                    result["wer"] < baseline["wer"]
                    and result["bins"]["tail"]["micro_recall"]
                    >= baseline["bins"]["tail"]["micro_recall"]
                    and result["overall"]["micro_f1"]
                    >= baseline["overall"]["micro_f1"]
                )
                candidates.append(result)
    eligible = [result for result in candidates if result["eligible"]]
    selected = min(
        eligible,
        key=lambda result: (
            result["wer"],
            -result["bins"]["tail"]["micro_recall"],
            -result["overall"]["micro_f1"],
            result["temperature"],
            result["ctc_weight"],
            result["denominator"],
        ),
        default=None,
    )
    return {
        "selection_rule": (
            "minimum D_sel SCLITE WER among MBR settings that improve "
            "CTC-only decoding while not reducing tail recall or overall "
            "micro-F1"
        ),
        "baseline": baseline,
        "candidates": candidates,
        "selected": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest", required=True)
    parser.add_argument("--gloss-dict", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--dataset-config",
        default=str(ROOT / "configs" / "phoenix14.yaml"),
    )
    parser.add_argument(
        "--temperatures",
        default="0.5,1,2,5,10",
    )
    parser.add_argument(
        "--ctc-weights",
        default="0,0.01,0.02,0.05,0.1,0.2,0.5,1",
    )
    parser.add_argument("--denominators", default="max")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-output")
    parser.add_argument("--allow-nonstrict", action="store_true")
    args = parser.parse_args()

    with open(args.nbest, encoding="utf-8") as source:
        cache = json.load(source)
    if cache.get("split") != "dev":
        raise ValueError("n-best cache must be built from D_sel/Dev")
    if not cache.get("strict_split") and not args.allow_nonstrict:
        raise ValueError(
            "non-strict n-best cache requires --allow-nonstrict"
        )
    with open(args.split_manifest, encoding="utf-8") as source:
        manifest = json.load(source)
    if (
        cache.get("strict_split")
        and cache.get("split_sha256") != manifest["split_sha256"]
    ):
        raise ValueError("n-best split hash does not match manifest")

    gloss_dict = np.load(
        args.gloss_dict, allow_pickle=True
    ).item()
    train_info = load_split_info(
        str(Path(args.split_manifest).parent),
        "train",
        args.split_manifest,
    )
    train_counts = gloss_occurrence_counts(train_info, gloss_dict)
    num_classes = len(gloss_dict) + 1
    with open(args.dataset_config, encoding="utf-8") as source:
        dataset_info = yaml.load(source, Loader=yaml.FullLoader)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="mbr-sclite-",
        dir=str(output_path.parent),
    ) as temporary_directory:
        evaluator = make_sclite_evaluator(
            cache["records"],
            dataset_info,
            gloss_dict,
            temporary_directory,
        )
        result = tune_grid(
            cache["records"],
            parse_float_list(args.temperatures),
            parse_float_list(args.ctc_weights),
            [
                value.strip()
                for value in args.denominators.split(",")
                if value.strip()
            ],
            train_counts,
            num_classes,
            evaluator,
        )

    serializable = {
        "selection_rule": result["selection_rule"],
        "baseline": {
            key: value
            for key, value in result["baseline"].items()
            if key != "hypotheses"
        },
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key != "hypotheses"
            }
            for candidate in result["candidates"]
        ],
        "selected": (
            {
                key: value
                for key, value in result["selected"].items()
                if key != "hypotheses"
            }
            if result["selected"] is not None else None
        ),
        "nbest_path": str(Path(args.nbest).resolve()),
        "split_sha256": manifest["split_sha256"],
    }
    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(serializable, destination, indent=2, sort_keys=True)
        destination.write("\n")
    if args.prediction_output and result["selected"] is not None:
        write_predictions(
            args.prediction_output,
            cache["records"],
            result["selected"]["hypotheses"],
            gloss_dict,
        )
    print(json.dumps({
        "baseline_wer": result["baseline"]["wer"],
        "selected": serializable["selected"],
        "output": str(output_path.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
