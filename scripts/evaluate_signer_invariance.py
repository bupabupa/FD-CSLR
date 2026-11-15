import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dataset.dataloader_video import BaseFeeder
from dataset.splits import load_split_manifest


def import_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate signer leakage with a probe and signer swap."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-weights", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--dataset-config", default="configs/phoenix14.yaml")
    parser.add_argument("--cal-split", default="cal")
    parser.add_argument("--eval-split", default="sel")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=0.01)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path):
    with open(resolve_path(path), encoding="utf-8") as source:
        return yaml.safe_load(source)


def make_feeder(config, dataset_info, gloss_dict, split_name, source_mode):
    feeder_args = config.get("feeder_args", {})
    return BaseFeeder(
        prefix=dataset_info["dataset_root"],
        gloss_dict=gloss_dict,
        drop_ratio=feeder_args.get("drop_ratio", 1),
        num_gloss=feeder_args.get("num_gloss", -1),
        mode=source_mode,
        transform_mode=False,
        datatype=feeder_args.get("datatype", "video"),
        split_manifest=str(resolve_path(config["split_manifest"])),
        split_name=split_name,
    )


def make_loader(feeder, args):
    return DataLoader(
        feeder,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=feeder.collate_fn,
    )


def build_model(config, dataset_info):
    gloss_dict = np.load(
        resolve_path(dataset_info["dict_path"]), allow_pickle=True
    ).item()
    model_args = dict(config["model_args"])
    model_args["num_classes"] = len(gloss_dict) + 1
    model_class = import_class(config["model"])
    model = model_class(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=config["loss_weights"],
    )
    return model, gloss_dict


def load_checkpoint(model, path, split_manifest):
    checkpoint = torch.load(resolve_path(path), map_location="cpu")
    expected = load_split_manifest(
        str(resolve_path(split_manifest)),
        str(resolve_path(split_manifest).parent),
    )["split_sha256"]
    if checkpoint.get("split_sha256") != expected:
        raise ValueError("checkpoint split hash does not match split manifest")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return expected


def forward_batch(model, video, video_length, signer_ids):
    kwargs = {}
    signature = inspect.signature(model.forward).parameters
    if "decode" in signature:
        kwargs["decode"] = False
    if getattr(model, "supports_signer", False):
        kwargs["signer_ids"] = signer_ids
    return model(video, video_length, **kwargs)


def masked_mean(sequence, lengths):
    time_steps = sequence.size(0)
    mask = (
        torch.arange(time_steps, device=sequence.device).view(-1, 1, 1)
        < lengths.to(sequence.device).view(1, -1, 1)
    )
    total = (sequence * mask).sum(dim=0)
    return total / lengths.to(sequence.device).clamp_min(1).view(-1, 1)


@torch.no_grad()
def collect_emission_features(model, loader, device, max_batches=0):
    features = []
    labels = []
    for batch_index, data in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        video = data[0].to(device)
        video_length = data[1].to(device)
        signer_ids = data[4].to(device)
        output = forward_batch(model, video, video_length, signer_ids)
        pooled = masked_mean(output["sequence_logits"], output["feat_len"])
        features.append(pooled.cpu())
        labels.append(data[4].cpu())
    if not features:
        raise RuntimeError("no samples were collected")
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def macro_recall(predictions, targets, num_classes):
    recalls = []
    for class_id in range(num_classes):
        selected = targets == class_id
        if selected.any():
            recalls.append(
                (predictions[selected] == class_id).float().mean().item()
            )
    return float(np.mean(recalls)) if recalls else 0.0


def fit_signer_probe(
    train_features,
    train_signers,
    eval_features,
    eval_signers,
    epochs,
    learning_rate,
    weight_decay,
):
    train_valid = train_signers > 0
    eval_valid = eval_signers > 0
    train_features = train_features[train_valid]
    train_signers = train_signers[train_valid]
    eval_features = eval_features[eval_valid]
    eval_signers = eval_signers[eval_valid]
    if train_features.numel() == 0 or eval_features.numel() == 0:
        return {"available": False, "reason": "missing signer ids"}

    train_ids = sorted(int(item) for item in train_signers.unique())
    eval_ids = sorted(int(item) for item in eval_signers.unique())
    if len(train_ids) < 2 or not set(eval_ids).issubset(set(train_ids)):
        return {
            "available": False,
            "reason": "probe requires overlapping train/eval signer ids",
            "train_signers": train_ids,
            "eval_signers": eval_ids,
        }

    mapping = {signer: index for index, signer in enumerate(train_ids)}
    train_targets = torch.tensor(
        [mapping[int(item)] for item in train_signers], dtype=torch.long
    )
    eval_targets = torch.tensor(
        [mapping[int(item)] for item in eval_signers], dtype=torch.long
    )
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_features = (train_features - mean) / std
    eval_features = (eval_features - mean) / std

    probe = nn.Linear(train_features.size(1), len(train_ids))
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(probe(train_features), train_targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = probe(eval_features)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == eval_targets).float().mean().item()
        nll = F.cross_entropy(logits, eval_targets).item()
    counts = torch.bincount(eval_targets, minlength=len(train_ids))
    majority_accuracy = counts.max().float().item() / counts.sum().item()
    return {
        "available": True,
        "train_samples": int(train_features.size(0)),
        "eval_samples": int(eval_features.size(0)),
        "train_signers": train_ids,
        "accuracy": float(accuracy),
        "macro_recall": macro_recall(
            predictions, eval_targets, len(train_ids)
        ),
        "majority_accuracy": float(majority_accuracy),
        "accuracy_minus_majority": float(accuracy - majority_accuracy),
        "nll": float(nll),
        "chance_accuracy": 1.0 / len(train_ids),
    }


def next_signer_map(model, loader, device):
    if not getattr(model, "supports_signer", False):
        return {}
    seen = model.signer_seen.nonzero(as_tuple=False).view(-1)
    seen = [int(item) for item in seen if int(item) > 0]
    if len(seen) < 2:
        selected = []
        for data in loader:
            selected.extend(
                int(item) for item in data[4].unique() if int(item) > 0
            )
        seen = sorted(set(selected))
    if len(seen) < 2:
        return {}
    seen = sorted(seen)
    return {
        signer: seen[(index + 1) % len(seen)]
        for index, signer in enumerate(seen)
    }


@torch.no_grad()
def evaluate_signer_swap(model, loader, device, signer_map, max_batches=0):
    if not signer_map:
        return {
            "available": False,
            "reason": "fewer than two seen signers",
        }
    total_kl = 0.0
    total_js = 0.0
    total_l1 = 0.0
    total_frames = 0
    for batch_index, data in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        video = data[0].to(device)
        video_length = data[1].to(device)
        original_signers = data[4].to(device)
        swapped = torch.tensor(
            [signer_map.get(int(item), int(item)) for item in original_signers],
            dtype=torch.long,
            device=device,
        )
        original = forward_batch(
            model, video, video_length, original_signers
        )["sequence_logits"]
        auxiliary = forward_batch(
            model, video, video_length, swapped
        )["sequence_logits"]
        original_log = original.log_softmax(dim=-1)
        auxiliary_log = auxiliary.log_softmax(dim=-1)
        original_probability = original_log.exp()
        auxiliary_probability = auxiliary_log.exp()
        mixture_log = (
            (original_probability + auxiliary_probability) * 0.5
        ).clamp_min(1e-8).log()
        valid = (
            torch.arange(original.size(0), device=device).view(1, -1)
            < video_length.view(-1, 1)
        ).unsqueeze(-1).expand_as(original)
        total_kl += (
            original_probability
            * (original_log - auxiliary_log)
        )[valid].sum().item()
        total_js += 0.5 * (
            original_probability * (original_log - mixture_log)
            + auxiliary_probability * (auxiliary_log - mixture_log)
        )[valid].sum().item()
        total_l1 += (
            original_probability - auxiliary_probability
        ).abs()[valid].sum().item()
        total_frames += int(valid.sum().item())
    if total_frames == 0:
        return {"available": False, "reason": "no valid frames"}
    return {
        "available": True,
        "frames": total_frames,
        "signer_rotation": {
            str(key): value for key, value in sorted(signer_map.items())
        },
        "mean_kl": total_kl / total_frames,
        "mean_js": total_js / total_frames,
        "mean_l1": total_l1 / total_frames,
    }


def main():
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("split_manifest") is None:
        config["split_manifest"] = args.split_manifest
    dataset_info = load_yaml(args.dataset_config)
    model, gloss_dict = build_model(config, dataset_info)
    split_sha256 = load_checkpoint(
        model, args.load_weights, args.split_manifest
    )
    device = torch.device(
        "cpu" if args.device < 0 else "cuda:{}".format(args.device)
    )
    model = model.to(device).eval()
    if (
        "decode" not in inspect.signature(model.forward).parameters
        and getattr(model, "decoder", None) is not None
    ):
        # Baseline prints are not needed for feature/probe/swap diagnostics.
        model.decoder.decode = lambda *args, **kwargs: None

    source_modes = {"train": "train", "cal": "train", "sel": "dev", "dev": "dev"}
    cal_feeder = make_feeder(
        config,
        dataset_info,
        gloss_dict,
        args.cal_split,
        source_modes[args.cal_split],
    )
    eval_feeder = make_feeder(
        config,
        dataset_info,
        gloss_dict,
        args.eval_split,
        source_modes[args.eval_split],
    )
    cal_loader = make_loader(cal_feeder, args)
    eval_loader = make_loader(eval_feeder, args)

    cal_features, cal_signers = collect_emission_features(
        model, cal_loader, device, args.max_batches
    )
    eval_features, eval_signers = collect_emission_features(
        model, eval_loader, device, args.max_batches
    )
    probe = fit_signer_probe(
        cal_features,
        cal_signers,
        eval_features,
        eval_signers,
        args.probe_epochs,
        args.probe_lr,
        args.probe_weight_decay,
    )
    signer_map = next_signer_map(model, eval_loader, device)
    swap = evaluate_signer_swap(
        model, eval_loader, device, signer_map, args.max_batches
    )
    result = {
        "config": str(resolve_path(args.config)),
        "checkpoint": str(resolve_path(args.load_weights)),
        "split_manifest": str(resolve_path(args.split_manifest)),
        "split_sha256": split_sha256,
        "cal_split": args.cal_split,
        "eval_split": args.eval_split,
        "max_batches": args.max_batches,
        "signer_probe": probe,
        "signer_swap": swap,
    }
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
