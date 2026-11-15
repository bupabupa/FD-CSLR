import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils
from dataset.splits import load_split_info, load_split_manifest
from evaluation.frequency_metrics import (
    class_frequency_report,
    gloss_occurrence_counts,
    sequence_error_counts,
)
from evaluation.slr_eval.wer_calculation import evaluate
from modules.calibration import (
    FUSION_PARAMETERIZATION_VERSION,
    validate_fusion_state,
)
from seq_scripts import write2file


def parse_member(spec, base_dir=None):
    fields = {}
    for item in spec.split(";"):
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError(
                "member fields must use key=value syntax: {}".format(spec)
            )
        if key in fields:
            raise ValueError("duplicate member field: {}".format(key))
        fields[key] = value

    required = {"name", "config", "checkpoint"}
    allowed = required | {"weight", "temperature"}
    missing = required.difference(fields)
    unexpected = set(fields).difference(allowed)
    if missing:
        raise ValueError(
            "member is missing fields: {}".format(sorted(missing))
        )
    if unexpected:
        raise ValueError(
            "member has unknown fields: {}".format(sorted(unexpected))
        )

    base_dir = Path(base_dir).expanduser().resolve() if base_dir else None

    def resolve_member_path(value):
        path = Path(value).expanduser()
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        return path.resolve()

    member = {
        "name": fields["name"],
        "config": resolve_member_path(fields["config"]),
        "checkpoint": resolve_member_path(fields["checkpoint"]),
        "weight": float(fields.get("weight", "1.0")),
        "temperature": float(fields.get("temperature", "1.0")),
    }
    if not member["name"]:
        raise ValueError("member name must not be empty")
    if member["weight"] < 0.0:
        raise ValueError("member weight must be non-negative")
    if member["temperature"] <= 0.0:
        raise ValueError("member temperature must be positive")
    if not member["config"].is_file():
        raise ValueError(
            "member config does not exist: {}".format(member["config"])
        )
    if not member["checkpoint"].is_file():
        raise ValueError(
            "member checkpoint does not exist: {}".format(
                member["checkpoint"]
            )
        )
    return member


def validate_members(members):
    if not members:
        raise ValueError("at least one ensemble member is required")
    names = [member["name"] for member in members]
    if len(names) != len(set(names)):
        raise ValueError("ensemble member names must be unique")
    if sum(member["weight"] for member in members) <= 0.0:
        raise ValueError("at least one ensemble member must have positive weight")


def load_ensemble_spec(path):
    path = Path(path).expanduser().resolve()
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("ensemble spec must contain a JSON object")
    raw_members = payload.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("ensemble spec must contain a non-empty members list")
    members = []
    for index, item in enumerate(raw_members):
        if not isinstance(item, dict):
            raise ValueError(
                "ensemble spec member {} must be an object".format(index)
            )
        fields = {}
        for key in ("name", "config", "checkpoint", "weight", "temperature"):
            value = item.get(key)
            if value is None:
                if key in {"weight", "temperature"}:
                    continue
                raise ValueError(
                    "ensemble spec member {} is missing {}".format(index, key)
                )
            fields[key] = str(value)
        members.append(parse_member(
            ";".join(
                "{}={}".format(key, value) for key, value in fields.items()
            ),
            base_dir=path.parent,
        ))
    validate_members(members)
    class_prior = payload.get("class_prior")
    prior_path = None
    prior_tau = 0.0
    if class_prior is not None:
        if not isinstance(class_prior, dict):
            raise ValueError("class_prior must be an object or null")
        prior_path = class_prior.get("path")
        prior_tau = class_prior.get("tau", 0.0)
        if prior_path is None:
            raise ValueError("class_prior path is required")
        prior_path = Path(prior_path).expanduser()
        if not prior_path.is_absolute():
            prior_path = path.parent / prior_path
        prior_path = prior_path.resolve()
        if not prior_path.is_file():
            raise ValueError(
                "class prior does not exist: {}".format(prior_path)
            )
        expected_hash = class_prior.get("sha256")
        if expected_hash and file_sha256(prior_path) != expected_hash:
            raise ValueError("class prior hash does not match ensemble spec")
        prior_tau = float(prior_tau)
        if prior_tau < 0.0:
            raise ValueError("class prior tau must be non-negative")
    return {
        "path": path,
        "sha256": file_sha256(path),
        "members": members,
        "class_prior_path": prior_path,
        "prior_tau": prior_tau,
        "split_sha256": payload.get("split_sha256"),
    }


def find_test_access_claim(output_dir):
    output_dir = Path(output_dir).resolve()
    for parent in (output_dir, *output_dir.parents):
        claim_path = parent / "test_access_claim.json"
        if not claim_path.is_file():
            continue
        with open(claim_path, encoding="utf-8") as source:
            claim = json.load(source)
        if not isinstance(claim, dict) or not isinstance(
            claim.get("selected"), dict
        ):
            raise ValueError(
                "invalid Test access claim: {}".format(claim_path)
            )
        required = {"name", "dev_prediction", "config", "checkpoint"}
        missing = required.difference(claim["selected"])
        if missing:
            raise ValueError(
                "invalid Test access claim: missing {}".format(
                    sorted(missing)
                )
            )
        return claim_path
    raise RuntimeError(
        "Test evaluation is locked: no test_access_claim.json was found "
        "in the output directory or its ancestors"
    )


def combine_probabilities(logits_list, members):
    if len(logits_list) != len(members):
        raise ValueError("one logits tensor is required for every member")
    combined = None
    total_weight = 0.0
    for logits, member in zip(logits_list, members):
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "{} produced non-finite logits".format(member["name"])
            )
        probabilities = (
            logits / float(member["temperature"])
        ).softmax(dim=-1)
        weighted = float(member["weight"]) * probabilities
        combined = weighted if combined is None else combined + weighted
        total_weight += float(member["weight"])
    combined = combined / total_weight
    return combined / combined.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def apply_class_prior_adjustment(probabilities, prior, prior_tau):
    """Boost rare non-blank classes using inverse training-frequency prior."""
    prior_tau = float(prior_tau)
    if prior_tau < 0.0:
        raise ValueError("prior_tau must be non-negative")
    if prior_tau == 0.0:
        return probabilities
    prior = torch.as_tensor(
        prior,
        dtype=probabilities.dtype,
        device=probabilities.device,
    ).flatten()
    if prior.numel() != probabilities.size(-1):
        raise ValueError("class prior must have one value per output class")
    if not torch.isfinite(prior).all() or (prior < 0).any():
        raise ValueError("class prior must contain finite non-negative values")
    factors = torch.ones_like(prior)
    nonblank_prior = prior[1:].clamp_min(torch.finfo(prior.dtype).tiny)
    factors[1:] = nonblank_prior.pow(-prior_tau)
    adjusted = probabilities * factors.view(1, 1, -1)
    return adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_yaml(path):
    with open(path, encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError("configuration must contain a mapping: {}".format(path))
    return value


def build_args(config_path, device, batch_size, num_workers, beam_width):
    config = load_yaml(config_path)
    parser = utils.get_parser()
    parser.set_defaults(**config)
    args = parser.parse_args([])
    args.config = str(config_path)
    args.device = str(device)
    args.phase = "test"
    args.test_batch_size = int(batch_size)
    args.num_worker = int(num_workers)
    args.beam_width = int(beam_width)
    args.eval_splits = ["dev"]
    dataset_info_path = ROOT / "configs" / "{}.yaml".format(args.dataset)
    args.dataset_info = load_yaml(dataset_info_path)
    return args


def load_model(
    member,
    args,
    gloss_dict,
    device,
    allow_unbound_checkpoints=False,
):
    state = torch.load(member["checkpoint"], map_location="cpu")
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(
            "checkpoint has no model_state_dict: {}".format(
                member["checkpoint"]
            )
        )
    if args.split_manifest is not None and not allow_unbound_checkpoints:
        expected = load_split_manifest(
            args.split_manifest,
            os.path.dirname(os.path.abspath(args.split_manifest)),
        )["split_sha256"]
        if state.get("split_sha256") != expected:
            raise ValueError(
                "checkpoint split hash does not match manifest: {}".format(
                    member["checkpoint"]
                )
            )

    prefix = "multichannel_calibrator."
    fusion_state = {
        name[len(prefix):]: value
        for name, value in model_state.items()
        if name.startswith(prefix)
    }
    if fusion_state:
        if (
            state.get("multichannel_parameterization_version")
            != FUSION_PARAMETERIZATION_VERSION
        ):
            raise ValueError(
                "multichannel checkpoint metadata version is missing or "
                "incompatible: {}".format(member["checkpoint"])
            )
        validate_fusion_state(fusion_state)

    model_class = import_class(args.model)
    model_args = dict(args.model_args)
    model_args["num_classes"] = len(gloss_dict) + 1
    model = model_class(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=args.loss_weights,
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_loader(args, split):
    feeder_class = import_class(args.feeder)
    feeder_args = dict(args.feeder_args)
    split_name = {
        "cal": "cal",
        "dev": "sel",
        "test": "test",
    }[split]
    feeder_args.update({
        "prefix": args.dataset_info["dataset_root"],
        "mode": "dev" if split in {"cal", "dev"} else "test",
        "transform_mode": False,
    })
    if args.split_manifest is not None:
        feeder_args["split_manifest"] = args.split_manifest
        feeder_args["split_name"] = split_name
    dataset = feeder_class(gloss_dict=args.gloss_dict, **feeder_args)
    loader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_worker,
        collate_fn=feeder_class.collate_fn,
    )
    return dataset, loader


def final_member_logits(model, ret_dict):
    logits = ret_dict["sequence_logits"]
    input_is_log_probs = ret_dict.get("sequence_log_probs") is not None
    if getattr(model, "multichannel_calibrator", None) is None and hasattr(
        model, "_prior_adjusted_logits"
    ):
        logits = model._prior_adjusted_logits(logits)
    if getattr(model, "fdgc_enabled", False):
        calibrated = model.fdgc(
            logits,
            source_prior=model._nonblank_source_prior(),
            difficulty=model.fdgc_difficulty,
            input_is_log_probs=input_is_log_probs,
        )
        logits = calibrated["logits"]
    return logits


def evaluate_predictions(
    args,
    output_dir,
    split,
    loader,
    info,
    metadata,
    reference_sequences,
    hypothesis_sequences,
    sentences,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if split == "cal":
        edits = 0
        reference_length = 0
        for reference, hypothesis in zip(
            reference_sequences,
            hypothesis_sequences,
        ):
            sample_edits, sample_length = sequence_error_counts(
                reference,
                hypothesis,
                len(loader.dataset.dict) + 1,
            )
            edits += int(sample_edits)
            reference_length += int(sample_length)
        wer = 100.0 * edits / max(reference_length, 1)
        return write_evaluation_artifacts(
            output_dir,
            split,
            loader,
            info,
            metadata,
            reference_sequences,
            hypothesis_sequences,
            sentences,
            wer,
            args,
        )
    work_dir = str(output_dir) + os.sep
    mode = split
    epoch = 6667
    evaluate_tool = args.evaluate_tool
    python_eval = evaluate_tool == "python"

    conv_sentences = sentences
    write2file(
        work_dir + "output-hypothesis-{}.ctm".format(mode),
        info,
        sentences,
    )
    write2file(
        work_dir + "output-hypothesis-{}-conv.ctm".format(mode),
        info,
        conv_sentences,
    )
    evaluate(
        prefix=work_dir,
        mode=mode,
        output_file="output-hypothesis-{}-conv.ctm".format(mode),
        evaluate_dir=args.dataset_info["evaluation_dir"],
        evaluate_prefix=args.dataset_info["evaluation_prefix"],
        output_dir="epoch_{}_{}_result/".format(epoch, mode),
        python_evaluate=python_eval,
    )
    wer = evaluate(
        prefix=work_dir,
        mode=mode,
        output_file="output-hypothesis-{}.ctm".format(mode),
        evaluate_dir=args.dataset_info["evaluation_dir"],
        evaluate_prefix=args.dataset_info["evaluation_prefix"],
        output_dir="epoch_{}_{}_result/".format(epoch, mode),
        python_evaluate=python_eval,
        triplet=True,
    )

    return write_evaluation_artifacts(
        output_dir,
        split,
        loader,
        info,
        metadata,
        reference_sequences,
        hypothesis_sequences,
        sentences,
        wer,
        args,
        epoch=epoch,
    )


def write_evaluation_artifacts(
    output_dir,
    split,
    loader,
    info,
    metadata,
    reference_sequences,
    hypothesis_sequences,
    sentences,
    wer,
    args,
    epoch=6667,
):
    output_dir = Path(output_dir)
    if args.split_manifest is not None:
        split_manifest = os.path.abspath(args.split_manifest)
        train_info = load_split_info(
            os.path.dirname(split_manifest),
            "train",
            split_manifest,
        )
    else:
        dict_dir = os.path.dirname(args.dataset_info["dict_path"])
        train_info = np.load(
            os.path.join(dict_dir, "train_info.npy"),
            allow_pickle=True,
        ).item()
    train_counts = gloss_occurrence_counts(train_info, loader.dataset.dict)
    report = class_frequency_report(
        reference_sequences,
        hypothesis_sequences,
        train_counts,
        blank_id=0,
    )
    report["decode"] = {
        "mode": split,
        "epoch": epoch,
        "evaluate_tool": args.evaluate_tool,
        "ensemble": True,
    }
    with open(
        output_dir / "frequency-metrics-{}.json".format(split),
        "w",
        encoding="utf-8",
    ) as destination:
        json.dump(report, destination, indent=2, sort_keys=True)

    id_to_gloss = {
        int(value[0]): str(gloss)
        for gloss, value in loader.dataset.dict.items()
    }
    with open(
        output_dir / "predictions-{}.jsonl".format(split),
        "w",
        encoding="utf-8",
    ) as destination:
        for sample_id, sample_metadata, reference, hypothesis, sentence in zip(
            info,
            metadata,
            reference_sequences,
            hypothesis_sequences,
            sentences,
        ):
            metadata_fields = sample_metadata.split("|")
            destination.write(json.dumps({
                "sample_id": sample_id,
                "signer": (
                    metadata_fields[2]
                    if len(metadata_fields) > 2
                    else ""
                ),
                "reference_ids": reference,
                "hypothesis_ids": hypothesis,
                "reference_glosses": [
                    id_to_gloss[class_id] for class_id in reference
                ],
                "hypothesis_glosses": [str(token[0]) for token in sentence],
            }, ensure_ascii=False, sort_keys=True))
            destination.write("\n")
    with open(
        output_dir / "{}.txt".format(split),
        "w",
        encoding="utf-8",
    ) as destination:
        destination.write(
            "Epoch {}, {} {:05.2f}%\n".format(epoch, split, wer)
        )
    return float(wer)


def decode_gloss(token):
    if isinstance(token, (tuple, list)):
        return str(token[0])
    return str(token)


def run(args, members, output_dir):
    if args.split == "test":
        find_test_access_claim(output_dir)

    config_args = build_args(
        members[0]["config"],
        args.device,
        args.batch_size,
        args.num_workers,
        args.beam_width,
    )
    gloss_dict = np.load(
        config_args.dataset_info["dict_path"],
        allow_pickle=True,
    ).item()
    config_args.gloss_dict = gloss_dict
    config_args.evaluate_tool = args.evaluate_tool
    config_args.split_manifest = (
        str(Path(args.split_manifest).expanduser().resolve())
        if args.split_manifest is not None
        else None
    )
    if args.ensemble_spec is not None:
        spec = load_ensemble_spec(args.ensemble_spec)
        if (
            config_args.split_manifest is not None
            and spec["split_sha256"]
            != load_split_manifest(
                config_args.split_manifest,
                os.path.dirname(os.path.abspath(config_args.split_manifest)),
            )["split_sha256"]
        ):
            raise ValueError(
                "ensemble spec split hash does not match the split manifest"
            )
    dataset, loader = build_loader(config_args, args.split)
    class_prior = None
    if args.class_prior_path is not None:
        class_prior = np.load(args.class_prior_path)
        if class_prior.shape != (len(gloss_dict) + 1,):
            raise ValueError(
                "class prior shape does not match the classifier output"
            )
    device = torch.device(
        "cuda:{}".format(args.device)
        if torch.cuda.is_available()
        else "cpu"
    )
    models = [
        load_model(
            member,
            build_args(
                member["config"],
                args.device,
                args.batch_size,
                args.num_workers,
                args.beam_width,
            ),
            gloss_dict,
            device,
            args.allow_unbound_checkpoints,
        )
        for member in members
    ]
    decoder = utils.Decode(
        gloss_dict,
        len(gloss_dict) + 1,
        "beam",
        beam_width=args.beam_width,
        num_processes=args.num_processes,
    )
    gloss_to_id = {
        str(gloss): int(value[0])
        for gloss, value in dataset.dict.items()
    }

    info = []
    metadata = []
    reference_sequences = []
    hypothesis_sequences = []
    sentences = []
    for batch in tqdm(loader):
        video = batch[0]
        video_length = batch[1]
        label = batch[2]
        label_length = batch[3]
        base = {
            "video": video,
            "video_length": video_length,
            "label": label.to(device),
            "label_lgt": label_length.to(device),
        }
        if len(batch) > 4:
            base["signer_ids"] = batch[4].to(device)

        member_logits = []
        feat_len = None
        conv_logits = None
        for model, member in zip(models, members):
            output = decode_member_flattened(model, base, device)
            current_len = output["feat_len"]
            if feat_len is None:
                feat_len = current_len
                conv_logits = output["conv_logits"]
            elif not torch.equal(feat_len, current_len):
                raise ValueError(
                    "ensemble members produced different frame lengths"
                )
            member_logits.append(final_member_logits(model, output))

        probabilities = combine_probabilities(member_logits, members)
        probabilities = apply_class_prior_adjustment(
            probabilities,
            class_prior if class_prior is not None else np.zeros(
                len(gloss_dict) + 1,
                dtype=np.float32,
            ),
            args.prior_tau,
        )
        batch_sentences = decoder.decode(
            probabilities,
            feat_len,
            batch_first=False,
            probs=True,
        )
        sentences.extend(batch_sentences)
        info.extend(item.split("|")[0] for item in batch[-1])
        metadata.extend(str(item) for item in batch[-1])

        label_offset = 0
        flat_labels = label.detach().cpu().tolist()
        for sample_index, length in enumerate(
            label_length.detach().cpu().tolist()
        ):
            label_end = label_offset + int(length)
            reference = flat_labels[label_offset:label_end]
            reference_sequences.append(reference)
            hypothesis = []
            for token in batch_sentences[sample_index]:
                gloss_id = gloss_to_id.get(decode_gloss(token), -1)
                if gloss_id >= 0:
                    hypothesis.append(gloss_id)
            hypothesis_sequences.append(hypothesis)
            label_offset = label_end

    wer = evaluate_predictions(
        config_args,
        output_dir,
        args.split,
        loader,
        info,
        metadata,
        reference_sequences,
        hypothesis_sequences,
        sentences,
    )
    all_member_specs = [
        {
            **member,
            "config": str(member["config"]),
            "checkpoint": str(member["checkpoint"]),
        }
        for member in members
    ]
    with open(
        Path(output_dir) / "ensemble_manifest.json",
        "w",
        encoding="utf-8",
    ) as destination:
        json.dump({
            "split": args.split,
            "members": all_member_specs,
            "wer": wer,
            "class_prior": (
                {
                    "path": str(
                        Path(args.class_prior_path).expanduser().resolve()
                    ),
                    "sha256": file_sha256(args.class_prior_path),
                    "tau": float(args.prior_tau),
                }
                if args.class_prior_path is not None
                else None
            ),
            "ensemble_spec": (
                {
                    "path": str(Path(args.ensemble_spec).expanduser().resolve()),
                    "sha256": file_sha256(args.ensemble_spec),
                }
                if args.ensemble_spec is not None
                else None
            ),
            "split_sha256": (
                load_split_manifest(
                    config_args.split_manifest,
                    os.path.dirname(config_args.split_manifest),
                )["split_sha256"]
                if config_args.split_manifest is not None
                else None
            ),
        }, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print("ensemble {} WER: {:.2f}%".format(args.split, wer))
    return wer


def decode_member_flattened(model, batch, device):
    model_kwargs = {
        "label": batch["label"],
        "label_lgt": batch["label_lgt"],
    }
    if getattr(model, "supports_signer", False):
        model_kwargs["signer_ids"] = batch["signer_ids"]
    if "decode" in inspect.signature(model.forward).parameters:
        model_kwargs["decode"] = False
    with torch.no_grad():
        return model(
            batch["video"].to(device),
            batch["video_length"].to(device),
            **model_kwargs,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append")
    parser.add_argument("--ensemble-spec", default=None)
    parser.add_argument(
        "--split",
        choices=["cal", "dev", "test"],
        default="dev",
    )
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--num-processes", type=int, default=10)
    parser.add_argument("--evaluate-tool", choices=["sclite", "python"], default="sclite")
    parser.add_argument("--class-prior-path", default=None)
    parser.add_argument("--prior-tau", type=float, default=0.0)
    parser.add_argument(
        "--allow-unbound-checkpoints",
        action="store_true",
        help="diagnostic only: allow checkpoints without a split hash",
    )
    args = parser.parse_args()
    if args.ensemble_spec is not None:
        if args.member:
            parser.error("--member and --ensemble-spec are mutually exclusive")
        spec = load_ensemble_spec(args.ensemble_spec)
        args.member = []
        members = spec["members"]
        if args.class_prior_path is None:
            args.class_prior_path = (
                str(spec["class_prior_path"])
                if spec["class_prior_path"] is not None
                else None
            )
            args.prior_tau = spec["prior_tau"]
    elif args.member:
        members = [parse_member(spec) for spec in args.member]
        validate_members(members)
    else:
        parser.error("--member or --ensemble-spec is required")
    if args.class_prior_path is None and args.prior_tau != 0.0:
        parser.error("--prior-tau requires --class-prior-path")
    if args.class_prior_path is not None and args.prior_tau < 0.0:
        parser.error("--prior-tau must be non-negative")
    run(args, members, Path(args.output_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
