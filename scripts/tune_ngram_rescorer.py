import argparse
import json
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
from evaluation.slr_eval.wer_calculation import evaluate as evaluate_sclite
from utils.ngram_rescorer import NGramRescorer


def parse_float_list(value):
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("grid cannot be empty")
    return values


def choose_beam(beams, rescorer, alpha, beta):
    if not beams:
        raise ValueError("sample has no CTC beam candidates")
    return min(
        range(len(beams)),
        key=lambda index: (
            float(beams[index]["ctc_score"])
            - float(alpha) * rescorer.score(beams[index]["tokens"])
            - float(beta) * len(beams[index]["tokens"]),
            index,
        ),
    )


def selected_hypotheses(records, rescorer, alpha, beta):
    return [
        list(record["beams"][choose_beam(
            record["beams"],
            rescorer,
            alpha,
            beta,
        )]["tokens"])
        for record in records
    ]


def evaluate_selection(
    records,
    rescorer,
    alpha,
    beta,
    train_counts,
    num_classes,
):
    references = [
        list(map(int, record["reference_ids"]))
        for record in records
    ]
    hypotheses = selected_hypotheses(
        records,
        rescorer,
        alpha,
        beta,
    )
    report = class_frequency_report(
        references,
        hypotheses,
        train_counts,
        blank_id=0,
    )
    if len(train_counts) != int(num_classes):
        raise ValueError("train counts do not match num_classes")
    edits = (
        int(report["overall"]["support"])
        - int(report["overall"]["true_positive"])
    )
    reference_occurrences = int(report["overall"]["support"])
    selected_beam_ids = [
        choose_beam(record["beams"], rescorer, alpha, beta)
        for record in records
    ]
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "num_sentences": len(records),
        "edits": edits,
        "reference_occurrences": reference_occurrences,
        "wer": float(
            100.0
            * edits
            / max(reference_occurrences, 1)
        ),
        "overall": report["overall"],
        "bins": report["bins"],
        "selected_ctc_beam_counts": {
            str(index): int(selected_beam_ids.count(index))
            for index in sorted(set(selected_beam_ids))
        },
        "references": references,
        "hypotheses": hypotheses,
    }


def tune_grid(
    records,
    rescorer,
    alphas,
    betas,
    train_counts,
    num_classes,
    wer_evaluator=None,
):
    baseline = evaluate_selection(
        records,
        rescorer,
        alpha=0.0,
        beta=0.0,
        train_counts=train_counts,
        num_classes=num_classes,
    )
    if wer_evaluator is not None:
        baseline["python_edit_wer"] = baseline["wer"]
        baseline["wer"] = float(wer_evaluator(baseline["hypotheses"]))
        baseline["wer_source"] = "sclite"
    else:
        baseline["wer_source"] = "python-edit-distance"
    candidates = []
    for alpha in alphas:
        for beta in betas:
            result = evaluate_selection(
                records,
                rescorer,
                alpha,
                beta,
                train_counts,
                num_classes,
            )
            if wer_evaluator is not None:
                result["python_edit_wer"] = result["wer"]
                result["wer"] = float(wer_evaluator(result["hypotheses"]))
                result["wer_source"] = "sclite"
            else:
                result["wer_source"] = "python-edit-distance"
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
            abs(result["alpha"]),
            abs(result["beta"]),
        ),
        default=None,
    )
    return {
        "selection_rule": (
            "minimum D_sel Dev WER among settings that improve the "
            "CTC-only baseline while not reducing tail recall or overall "
            "micro-F1"
        ),
        "baseline": baseline,
        "candidates": candidates,
        "selected": selected,
    }


def make_sclite_evaluator(
    records,
    dataset_info,
    gloss_dict,
    directory,
):
    id_to_gloss = {
        int(value[0]): str(gloss)
        for gloss, value in gloss_dict.items()
    }
    counter = {"value": 0}

    def evaluate(hypotheses):
        index = counter["value"]
        counter["value"] += 1
        ctm_name = "ngram-{}.ctm".format(index)
        ctm_path = Path(directory) / ctm_name
        with open(ctm_path, "w", encoding="utf-8") as destination:
            for record, hypothesis in zip(records, hypotheses):
                for token_index, class_id in enumerate(hypothesis):
                    destination.write(
                        "{} 1 {:.2f} {:.2f} {}\n".format(
                            record["sample_id"],
                            token_index * 1.0 / 100,
                            (token_index + 1) * 1.0 / 100,
                            id_to_gloss[int(class_id)],
                        )
                    )
        return evaluate_sclite(
            prefix=str(Path(directory)) + "/",
            mode="dev",
            evaluate_dir=dataset_info["evaluation_dir"],
            evaluate_prefix=dataset_info["evaluation_prefix"],
            output_file=ctm_name,
            output_dir="report-{}/".format(index),
            python_evaluate=False,
        )

    return evaluate


def write_predictions(path, records, hypotheses, gloss_dict):
    id_to_gloss = {
        int(value[0]): str(gloss)
        for gloss, value in gloss_dict.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as destination:
        for record, hypothesis in zip(records, hypotheses):
            destination.write(json.dumps({
                "sample_id": record["sample_id"],
                "signer": record.get("signer", ""),
                "reference_ids": record["reference_ids"],
                "hypothesis_ids": hypothesis,
                "reference_glosses": [
                    id_to_gloss[class_id]
                    for class_id in record["reference_ids"]
                ],
                "hypothesis_glosses": [
                    id_to_gloss[class_id] for class_id in hypothesis
                ],
            }, ensure_ascii=False, sort_keys=True))
            destination.write("\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbest", required=True)
    parser.add_argument("--rescorer", required=True)
    parser.add_argument("--gloss-dict", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--dataset-config",
        default=str(ROOT / "configs" / "phoenix14.yaml"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-output")
    parser.add_argument(
        "--alphas",
        default="0,0.02,0.05,0.1,0.2",
    )
    parser.add_argument("--betas", default="0")
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
    if cache.get("strict_split"):
        if cache.get("split_sha256") != manifest["split_sha256"]:
            raise ValueError("n-best split hash does not match manifest")

    with open(args.rescorer, encoding="utf-8") as source:
        rescorer_payload = json.load(source)
    if rescorer_payload.get("source") != "D_train":
        raise ValueError("rescorer must be built only from D_train")
    if rescorer_payload.get("split_sha256") != manifest["split_sha256"]:
        raise ValueError("rescorer split hash does not match manifest")
    rescorer = NGramRescorer.from_json(args.rescorer)

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
    if len(train_counts) != num_classes:
        raise ValueError("training gloss counts do not match num_classes")

    with open(args.dataset_config, encoding="utf-8") as source:
        dataset_info = yaml.load(source, Loader=yaml.FullLoader)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ngram-sclite-",
        dir=str(output_path.parent),
    ) as temporary_directory:
        wer_evaluator = make_sclite_evaluator(
            cache["records"],
            dataset_info,
            gloss_dict,
            temporary_directory,
        )
        result = tune_grid(
            cache["records"],
            rescorer,
            parse_float_list(args.alphas),
            parse_float_list(args.betas),
            train_counts,
            num_classes,
            wer_evaluator=wer_evaluator,
        )
    serializable = {
        key: value
        for key, value in result.items()
        if key != "candidates"
    }
    serializable["candidates"] = []
    for candidate in result["candidates"]:
        serializable["candidates"].append({
            key: value
            for key, value in candidate.items()
            if key not in {"references", "hypotheses"}
        })
    serializable["baseline"] = {
        key: value
        for key, value in result["baseline"].items()
        if key not in {"references", "hypotheses"}
    }
    if result["selected"] is not None:
        serializable["selected"] = {
            key: value
            for key, value in result["selected"].items()
            if key not in {"references", "hypotheses"}
        }
    serializable["nbest_path"] = str(Path(args.nbest).resolve())
    serializable["rescorer_path"] = str(Path(args.rescorer).resolve())
    serializable["split_sha256"] = manifest["split_sha256"]
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
        "selected": serializable.get("selected"),
        "output": str(output_path.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
