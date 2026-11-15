import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.dataloader_video import BaseFeeder  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALLOWED_SPLITS = {"train", "cal", "sel", "test"}


def validate_split_access(splits, allow_test=False):
    splits = [str(value) for value in splits]
    unknown = set(splits).difference(ALLOWED_SPLITS)
    if unknown:
        raise ValueError(
            "unknown split names: {}".format(sorted(unknown))
        )
    if len(splits) != len(set(splits)):
        raise ValueError("split names must be unique")
    if "test" in splits and not allow_test:
        raise ValueError(
            "Test extraction is locked; pass --allow-test only after "
            "candidate locking"
        )
    return splits


def evenly_spaced_indices(size, maximum):
    size = int(size)
    maximum = int(maximum)
    if size < 1:
        raise ValueError("cannot sample frames from an empty video")
    if maximum < 1:
        raise ValueError("maximum sampled frames must be positive")
    count = min(size, maximum)
    indices = np.linspace(0, size - 1, num=count, dtype=np.int64)
    return indices.tolist()


def pool_frame_embeddings(features):
    """Average sampled-frame features and L2-normalize each clip."""
    features = torch.as_tensor(features)
    if features.ndim != 3:
        raise ValueError("features must have shape [B,F,D]")
    if not torch.isfinite(features).all():
        raise ValueError("frame features must be finite")
    pooled = features.mean(dim=1)
    return torch.nn.functional.normalize(pooled, dim=-1, eps=1e-6)


def _resolve(root, value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(root) / path).resolve()


def _feature_model(backbone, device):
    if backbone == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    elif backbone == "resnet34":
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet34(weights=weights)
    else:
        raise ValueError("backbone must be resnet18 or resnet34")
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


def _frame_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _frame_paths(dataset_root, record, maximum):
    folder = os.path.join(
        str(dataset_root),
        "features/fullFrame-256x256px",
        str(record["folder"]),
    )
    paths = sorted(glob.glob(folder))
    if not paths:
        raise FileNotFoundError(
            "no video frames found for sample {} at {}".format(
                record.get("fileid"),
                folder,
            )
        )
    return [
        paths[index]
        for index in evenly_spaced_indices(len(paths), maximum)
    ]


def _load_frame(path, transform):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("failed to read frame: {}".format(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(image))


def _split_hash(manifest_path, split):
    with open(manifest_path, encoding="utf-8") as source:
        manifest = json.load(source)
    return manifest["split_sha256"], manifest["splits"][split]


def extract(
    config_path,
    splits,
    output_path,
    backbone="resnet18",
    sampled_frames=16,
    device="0",
    limit=None,
    overwrite=False,
    root=ROOT,
):
    root = Path(root).resolve()
    splits = validate_split_access(splits)
    output_path = _resolve(root, output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "output exists; pass --overwrite to replace it: {}".format(
                output_path
            )
        )
    config_path = _resolve(root, config_path)
    with open(config_path, encoding="utf-8") as source:
        dataset_info = yaml.safe_load(source)
    dataset_root = _resolve(root, dataset_info["dataset_root"])
    gloss_dict = np.load(
        _resolve(root, dataset_info["dict_path"]),
        allow_pickle=True,
    ).item()
    split_manifest = root / "preprocess/phoenix2014/split_manifest.json"
    if not split_manifest.is_file():
        raise FileNotFoundError(
            "split manifest does not exist: {}".format(split_manifest)
        )

    if str(device) == "cpu" or not torch.cuda.is_available():
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cuda:{}".format(device))
    model = _feature_model(backbone, torch_device)
    transform = _frame_transform()

    fileids = []
    embeddings = []
    sample_splits = []
    split_hashes = {}
    for split in splits:
        split_hash, split_records = _split_hash(split_manifest, split)
        split_hashes[split] = split_hash
        feeder = BaseFeeder(
            prefix=str(dataset_root),
            gloss_dict=gloss_dict,
            drop_ratio=1.0,
            num_gloss=-1,
            mode=split,
            transform_mode=False,
            datatype="video",
            split_manifest=str(split_manifest),
            split_name=split,
        )
        record_count = len(feeder)
        if limit is not None:
            record_count = min(record_count, int(limit))
        for index in range(record_count):
            record = feeder.inputs_list[index]
            paths = _frame_paths(
                dataset_root,
                record,
                sampled_frames,
            )
            frames = torch.stack(
                [_load_frame(path, transform) for path in paths]
            ).to(torch_device)
            with torch.inference_mode():
                frame_features = model(frames)
                embedding = pool_frame_embeddings(
                    frame_features.unsqueeze(0)
                )[0]
            fileids.append(str(record["fileid"]))
            sample_splits.append(split)
            embeddings.append(embedding.cpu().numpy().astype(np.float32))
            if (index + 1) % 100 == 0 or index + 1 == record_count:
                print(
                    "{} {}/{}".format(split, index + 1, record_count),
                    flush=True,
                )

    if len(fileids) != len(set(fileids)):
        raise ValueError("duplicate fileids found across extracted splits")
    embedding_array = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(embedding_array).all():
        raise ValueError("extracted embeddings contain non-finite values")
    fileid_digest = hashlib.sha256(
        json.dumps(fileids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "backbone": backbone,
        "pretraining": "ImageNet1K_V1",
        "sampled_frames": int(sampled_frames),
        "input_resolution": 224,
        "pooling": "frame_mean_then_l2",
        "perspective": "deterministic center crop; no random augmentation",
        "splits": list(splits),
        "split_hashes": split_hashes,
        "fileids_sha256": fileid_digest,
        "fit_rule": "cluster only on train; assign cal/sel/test with fixed centers",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        fileids=np.asarray(fileids, dtype=str),
        embeddings=embedding_array,
        split=np.asarray(sample_splits, dtype=str),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(output_path)
    return {
        "output": str(output_path),
        "num_samples": len(fileids),
        "embedding_shape": list(embedding_array.shape),
        **metadata,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phoenix14.yaml")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backbone",
        choices=("resnet18", "resnet34"),
        default="resnet18",
    )
    parser.add_argument("--sampled-frames", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    result = extract(
        config_path=args.config,
        splits=validate_split_access(args.splits, args.allow_test),
        output_path=args.output,
        backbone=args.backbone,
        sampled_frames=args.sampled_frames,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
