import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils
from dataset.splits import load_split_manifest
from scripts.merge_fdgc_checkpoint import _load_checkpoint
from scripts.evaluate_logit_ensemble import (
    build_args,
    build_loader,
    decode_member_flattened,
    file_sha256,
    final_member_logits,
    load_model,
    parse_member,
    validate_members,
)
from sequence_normalization import (
    aggregate_wer,
    normalize_token_sequences,
)


TAU_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15)
TEMPERATURE_RANGE = (0.5, 2.0)
PREPROCESS_SCRIPT = ROOT / "evaluation" / "slr_eval" / "preprocess.sh"


def prepare_batch(batch, device):
    values = {
        "video": batch[0],
        "video_length": batch[1],
        "label": batch[2],
        "label_lgt": batch[3],
    }
    if len(batch) > 4:
        values["signer_ids"] = batch[4]
    return {
        "video": values["video"],
        "video_length": values["video_length"],
        "label": values["label"].to(device),
        "label_lgt": values["label_lgt"].to(device),
        "signer_ids": (
            values["signer_ids"].to(device)
            if values.get("signer_ids") is not None
            else None
        ),
    }


def collect_member_logits(
    member,
    args,
    loader,
    gloss_dict,
    device,
    allow_unbound_checkpoints=False,
    cache_dir=None,
):
    cache_path = None
    if cache_dir is not None:
        safe_name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in member["name"]
        )
        cache_path = Path(cache_dir) / "{}.pt".format(safe_name)
        if cache_path.is_file():
            try:
                payload = torch.load(
                    str(cache_path),
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception:
                payload = {}
            expected = {
                "checkpoint_sha256": file_sha256(member["checkpoint"]),
                "config_sha256": file_sha256(member["config"]),
                "split_sha256": load_split_manifest(
                    args.split_manifest,
                    os.path.dirname(os.path.abspath(args.split_manifest)),
                )["split_sha256"],
                "records": len(loader.dataset),
            }
            if (
                payload.get("metadata") == expected
                and len(payload.get("samples", [])) == len(loader.dataset)
            ):
                return {
                    "samples": payload["samples"],
                    "input_lengths": payload["input_lengths"],
                }
    member_args = build_args(
        member["config"],
        args.device,
        args.test_batch_size,
        args.num_worker,
        args.beam_width,
    )
    member_args.split_manifest = args.split_manifest
    member_args.gloss_dict = gloss_dict
    model = load_model(
        member,
        member_args,
        gloss_dict,
        device,
        allow_unbound_checkpoints=allow_unbound_checkpoints,
    )
    samples = []
    input_lengths = []
    for batch in tqdm(loader, desc="cal logits " + member["name"]):
        base = prepare_batch(batch, device)
        output = decode_member_flattened(model, base, device)
        logits = final_member_logits(model, output)
        lengths = output["feat_len"].detach().cpu().int().tolist()
        input_lengths.extend(int(length) for length in lengths)
        for index, length in enumerate(lengths):
            samples.append(
                logits[:int(length), index, :].detach().cpu().float().clone()
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    record = {
        "samples": samples,
        "input_lengths": input_lengths,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        torch.save({
            "metadata": {
                "checkpoint_sha256": file_sha256(member["checkpoint"]),
                "config_sha256": file_sha256(member["config"]),
                "split_sha256": load_split_manifest(
                    args.split_manifest,
                    os.path.dirname(os.path.abspath(args.split_manifest)),
                )["split_sha256"],
                "records": len(loader.dataset),
            },
            **record,
        }, temporary)
        os.replace(temporary, cache_path)
    return record


def collect_targets(loader):
    references = []
    lengths = []
    metadata = []
    labels = []
    for batch in loader:
        flat_labels = batch[2].detach().cpu().int().tolist()
        label_lengths = batch[3].detach().cpu().int().tolist()
        offset = 0
        for sample_index, length in enumerate(label_lengths):
            end = offset + int(length)
            reference = flat_labels[offset:end]
            references.append(reference)
            labels.extend(reference)
            lengths.append(int(length))
            metadata.append(str(batch[-1][sample_index]))
            offset = end
    return references, lengths, metadata, labels


def pad_member_logits(samples, device):
    return pad_sequence(
        [sample.to(device) for sample in samples],
        batch_first=False,
    )


def normalize_weights(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty vector")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    return values / total


def combine_padded_logits(member_logits, weights, temperatures, indices):
    combined = None
    total = 0.0
    for member_index, weight, temperature in zip(
        indices,
        weights,
        temperatures,
    ):
        weight = float(weight)
        if weight == 0.0:
            continue
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        probabilities = (
            member_logits[member_index] / float(temperature)
        ).softmax(dim=-1)
        weighted = weight * probabilities
        combined = weighted if combined is None else combined + weighted
        total += weight
    if combined is None or total <= 0.0:
        raise ValueError("combined ensemble needs at least one positive weight")
    combined = combined / total
    return combined / combined.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def apply_prior(probabilities, prior, tau, lengths):
    if tau == 0.0:
        return probabilities
    factors = torch.ones_like(prior)
    factors[1:] = prior[1:].clamp_min(torch.finfo(prior.dtype).tiny).pow(
        -float(tau)
    )
    adjusted = probabilities * factors.view(1, 1, -1)
    adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mask = (
        torch.arange(adjusted.size(0), device=adjusted.device)
        .view(-1, 1)
        < torch.as_tensor(lengths, device=adjusted.device).view(1, -1)
    )
    return adjusted * mask.unsqueeze(-1)


def candidate_ctc_nll(
    member_logits,
    weights,
    temperatures,
    indices,
    prior,
    tau,
    target,
    target_lengths,
    input_lengths,
):
    probabilities = combine_padded_logits(
        member_logits,
        weights,
        temperatures,
        indices,
    )
    probabilities = apply_prior(probabilities, prior, tau, input_lengths)
    log_probabilities = probabilities.clamp_min(1e-12).log()
    try:
        loss = F.ctc_loss(
            log_probabilities,
            target,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="mean",
            zero_infinity=False,
        )
    except RuntimeError:
        return float("inf")
    return float(loss.detach().cpu())


def subset_candidates(num_members, max_members):
    for size in range(1, min(num_members, max_members) + 1):
        for indices in itertools.combinations(range(num_members), size):
            yield indices


def random_refinements(base_subset, rng, count):
    indices = tuple(base_subset)
    for _ in range(count):
        weights = normalize_weights(rng.dirichlet(np.ones(len(indices))))
        temperatures = np.exp(rng.uniform(
            math.log(TEMPERATURE_RANGE[0]),
            math.log(TEMPERATURE_RANGE[1]),
            size=len(indices),
        ))
        yield (
            indices,
            tuple(float(value) for value in weights),
            tuple(float(value) for value in temperatures),
            0.0,
        )


def deterministic_anchor_specs(base_subset):
    indices = tuple(base_subset)
    if len(indices) == 1:
        weight_sets = [(1.0,)]
    elif len(indices) == 2:
        weight_sets = [
            (fraction, 1.0 - fraction)
            for fraction in (0.5, 0.75, 0.8, 0.9, 0.95)
        ]
    elif len(indices) == 3:
        weight_sets = [
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.6, 0.2, 0.2),
            (0.8, 0.1, 0.1),
        ]
    else:
        uniform = 1.0 / len(indices)
        weight_sets = [
            tuple(uniform for _ in indices),
            (0.7,) + tuple(0.1 for _ in indices[1:]),
        ]

    temperature_sets = [
        tuple(temperature for _ in indices)
        for temperature in (0.5, 0.75, 1.0, 1.25, 1.5)
    ]
    if len(indices) == 2:
        temperature_sets.extend([
            (0.75, 0.9),
            (0.9, 0.75),
            (0.75, 1.0),
            (1.0, 0.75),
        ])
    for weights in weight_sets:
        weights = tuple(float(value) for value in normalize_weights(weights))
        for temperatures in temperature_sets:
            for tau in TAU_GRID:
                yield weights, temperatures, tau


def config_key(record):
    return (
        round(float(record["cal_nll"]), 10),
        len(record["member_indices"]),
        round(float(record["prior_tau"]), 10),
        tuple(round(value, 10) for value in record["temperatures"]),
        tuple(round(value, 10) for value in record["weights"]),
    )


def decode_cal_candidate(
    member_logits,
    candidate,
    decoder,
    prior,
    input_lengths,
    references,
    gloss_to_id,
    normalized_references,
):
    probabilities = combine_padded_logits(
        member_logits,
        candidate["weights"],
        candidate["temperatures"],
        candidate["member_indices"],
    )
    probabilities = apply_prior(
        probabilities,
        prior,
        candidate["prior_tau"],
        input_lengths,
    )
    sentences = decoder.decode(
        probabilities.cpu(),
        torch.tensor(input_lengths, dtype=torch.int32),
        batch_first=False,
        probs=True,
    )
    hypotheses = []
    unknown = 0
    for sentence in sentences:
        hypothesis = []
        for token in sentence:
            gloss = str(token[0] if isinstance(token, (tuple, list)) else token)
            gloss_id = gloss_to_id.get(gloss, -1)
            if gloss_id < 0:
                unknown += 1
            hypothesis.append(gloss)
        hypotheses.append(hypothesis)
    normalized_hypotheses = normalize_token_sequences(
        hypotheses,
        PREPROCESS_SCRIPT,
    )
    metrics = aggregate_wer(
        normalized_references,
        normalized_hypotheses,
    )
    return {
        "cal_wer": metrics["wer"],
        "edit_distance": metrics["edits"],
        "reference_length": metrics["reference_length"],
        "unknown_hypothesis_tokens": unknown,
    }


def ensure_candidate(summary, candidate):
    record = {
        "member_indices": list(candidate["member_indices"]),
        "weights": list(candidate["weights"]),
        "temperatures": list(candidate["temperatures"]),
        "prior_tau": float(candidate["prior_tau"]),
        "cal_nll": float(candidate["cal_nll"]),
    }
    summary.append(record)
    return record


def select_candidate(
    members,
    args,
    loader,
    base_args,
    gloss_dict,
    device,
    split_sha256,
):
    references, target_lengths, metadata, flat_labels = collect_targets(loader)
    member_records = [
        collect_member_logits(
            member,
            base_args,
            loader,
            gloss_dict,
            device,
            args.allow_unbound_checkpoints,
            args.cache_dir,
        )
        for member in members
    ]
    input_lengths = member_records[0]["input_lengths"]
    for member, record in zip(members, member_records):
        if record["input_lengths"] != input_lengths:
            raise ValueError(
                "{} produced different calibration frame lengths".format(
                    member["name"]
                )
            )
    member_logits_cpu = [record["samples"] for record in member_records]
    member_logits = [
        pad_member_logits(samples, device)
        for samples in member_logits_cpu
    ]
    target = torch.tensor(flat_labels, dtype=torch.long, device=device)
    target_lengths_tensor = torch.tensor(
        target_lengths,
        dtype=torch.long,
        device=device,
    )
    input_lengths_tensor = torch.tensor(
        input_lengths,
        dtype=torch.long,
        device=device,
    )
    if args.class_prior_path is None:
        prior = torch.zeros(
            len(gloss_dict) + 1,
            dtype=torch.float32,
            device=device,
        )
    else:
        prior = torch.as_tensor(
            np.load(args.class_prior_path),
            dtype=torch.float32,
            device=device,
        ).flatten()
    if prior.numel() != len(gloss_dict) + 1:
        raise ValueError("class prior shape does not match dictionary")

    evaluated = []
    anchors = []
    for indices in subset_candidates(len(members), args.max_members):
        weights = normalize_weights(np.ones(len(indices)))
        temperatures = np.ones(len(indices), dtype=np.float64)
        for tau in TAU_GRID:
            nll = candidate_ctc_nll(
                member_logits,
                tuple(weights),
                tuple(temperatures),
                indices,
                prior,
                tau,
                target,
                target_lengths_tensor,
                input_lengths_tensor,
            )
            if not math.isfinite(nll):
                continue
            ensure_candidate(evaluated, {
                "member_indices": indices,
                "weights": tuple(weights),
                "temperatures": tuple(temperatures),
                "prior_tau": tau,
                "cal_nll": nll,
            })
        for weights, temperatures, tau in deterministic_anchor_specs(indices):
            nll = candidate_ctc_nll(
                member_logits,
                weights,
                temperatures,
                indices,
                prior,
                tau,
                target,
                target_lengths_tensor,
                input_lengths_tensor,
            )
            if not math.isfinite(nll):
                continue
            record = ensure_candidate(anchors, {
                "member_indices": indices,
                "weights": weights,
                "temperatures": temperatures,
                "prior_tau": tau,
                "cal_nll": nll,
            })
            evaluated.append(record)
    evaluated.sort(key=config_key)
    top_subsets = []
    seen_subsets = set()
    for record in evaluated:
        subset = tuple(record["member_indices"])
        if subset in seen_subsets:
            continue
        seen_subsets.add(subset)
        top_subsets.append(subset)
        if len(top_subsets) >= args.refinement_subsets:
            break

    rng = np.random.default_rng(args.seed)
    for subset in top_subsets:
        for indices, weights, temperatures, _ in random_refinements(
            subset,
            rng,
            args.refinement_samples,
        ):
            for tau in TAU_GRID:
                nll = candidate_ctc_nll(
                    member_logits,
                    weights,
                    temperatures,
                    indices,
                    prior,
                    tau,
                    target,
                    target_lengths_tensor,
                    input_lengths_tensor,
                )
                if not math.isfinite(nll):
                    continue
                ensure_candidate(evaluated, {
                    "member_indices": indices,
                    "weights": weights,
                    "temperatures": temperatures,
                    "prior_tau": tau,
                    "cal_nll": nll,
                })
    evaluated.sort(key=config_key)

    per_subset = {}
    for record in evaluated:
        subset = tuple(record["member_indices"])
        per_subset.setdefault(subset, record)
    anchors.sort(key=config_key)
    shortlist = list(per_subset.values())
    shortlist.extend(anchors)
    shortlist.extend(evaluated)
    deduplicated = []
    seen = set()
    for record in shortlist:
        signature = (
            tuple(record["member_indices"]),
            tuple(round(value, 5) for value in record["weights"]),
            tuple(round(value, 5) for value in record["temperatures"]),
            round(record["prior_tau"], 5),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduplicated.append(record)
    decode_limit = max(
        args.decode_candidates,
        len(per_subset) + len(anchors),
    )
    shortlist = deduplicated[:decode_limit]

    gloss_to_id = {
        str(gloss): int(value[0])
        for gloss, value in loader.dataset.dict.items()
    }
    id_to_gloss = {
        int(value[0]): str(gloss)
        for gloss, value in loader.dataset.dict.items()
    }
    normalized_references = normalize_token_sequences(
        [
            [id_to_gloss[class_id] for class_id in reference]
            for reference in references
        ],
        PREPROCESS_SCRIPT,
    )
    decoder = utils.Decode(
        gloss_dict,
        len(gloss_dict) + 1,
        "beam",
        beam_width=args.beam_width,
        num_processes=args.num_processes,
    )
    for record in shortlist:
        metrics = decode_cal_candidate(
            member_logits,
            record,
            decoder,
            prior,
            input_lengths,
            references,
            gloss_to_id,
            normalized_references,
        )
        record.update(metrics)
    shortlist.sort(key=lambda record: (
        record["cal_wer"],
        record["cal_nll"],
        len(record["member_indices"]),
        record["prior_tau"],
    ))
    best = shortlist[0]
    selected_members = []
    for local_index, member_index in enumerate(best["member_indices"]):
        member = dict(members[member_index])
        member["config"] = str(member["config"])
        member["checkpoint"] = str(member["checkpoint"])
        member["weight"] = float(best["weights"][local_index])
        member["temperature"] = float(best["temperatures"][local_index])
        selected_members.append(member)
    return {
        "selection_rule": (
            "all subset, weighting, temperature and prior choices use D_cal; "
            "WER uses the VAC official preprocess normalization; "
            "D_sel is not read"
        ),
        "split_sha256": split_sha256,
        "members": selected_members,
        "class_prior_path": (
            str(Path(args.class_prior_path).resolve())
            if args.class_prior_path is not None
            else None
        ),
        "class_prior_sha256": (
            file_sha256(args.class_prior_path)
            if args.class_prior_path is not None
            else None
        ),
        "prior_tau": float(best["prior_tau"]),
        "cal_metrics": {
            key: value
            for key, value in best.items()
            if key not in {"member_indices", "weights", "temperatures"}
        },
        "cal_shortlist": shortlist,
        "search_size": len(evaluated),
        "cal_records": len(references),
        "cal_metadata_preview": metadata[:3],
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--class-prior-path", default=None)
    parser.add_argument("--output-spec", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-processes", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--max-members", type=int, default=3)
    parser.add_argument("--refinement-subsets", type=int, default=5)
    parser.add_argument("--refinement-samples", type=int, default=50)
    parser.add_argument("--decode-candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-unbound-checkpoints", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    members = [parse_member(spec) for spec in args.member]
    validate_members(members)
    split_manifest = Path(args.split_manifest).expanduser().resolve()
    split_sha256 = load_split_manifest(
        str(split_manifest),
        os.path.dirname(str(split_manifest)),
    )["split_sha256"]
    if not args.allow_unbound_checkpoints:
        for member in members:
            payload = _load_checkpoint(member["checkpoint"])
            if payload.get("split_sha256") != split_sha256:
                raise ValueError(
                    "member checkpoint is not bound to the strict split: {}".format(
                        member["checkpoint"]
                    )
                )
    device = torch.device(
        "cuda:{}".format(args.device)
        if torch.cuda.is_available()
        else "cpu"
    )
    base_args = build_args(
        members[0]["config"],
        args.device,
        args.batch_size,
        args.num_workers,
        args.beam_width,
    )
    base_args.split_manifest = str(split_manifest)
    gloss_dict = np.load(
        base_args.dataset_info["dict_path"],
        allow_pickle=True,
    ).item()
    base_args.gloss_dict = gloss_dict
    _, loader = build_loader(base_args, "cal")
    result = select_candidate(
        members,
        args,
        loader,
        base_args,
        gloss_dict,
        device,
        split_sha256,
    )
    spec = {
        "members": [
            {
                "name": member["name"],
                "config": str(member["config"]),
                "checkpoint": str(member["checkpoint"]),
                "weight": float(member["weight"]),
                "temperature": float(member["temperature"]),
            }
            for member in result["members"]
        ],
        "class_prior": (
            {
                "path": result["class_prior_path"],
                "sha256": result["class_prior_sha256"],
                "tau": result["prior_tau"],
            }
            if result["class_prior_path"] is not None
            else None
        ),
        "split_sha256": split_sha256,
    }
    write_json(args.output_spec, spec)
    write_json(args.output_summary, result)
    print(json.dumps(result["cal_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
