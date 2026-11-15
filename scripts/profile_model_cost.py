import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import get_parser  # noqa: E402


def temporal_frame_floor(conv_type):
    floors = {0: 3, 1: 6, 2: 16}
    try:
        return floors[int(conv_type)]
    except (KeyError, TypeError, ValueError):
        raise ValueError("conv_type must be one of 0, 1, or 2")


def validate_profile_request(
    conv_type,
    batch_size,
    frames,
    height,
    width,
    warmup,
    iterations,
):
    values = {
        "batch_size": int(batch_size),
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "warmup": int(warmup),
        "iterations": int(iterations),
    }
    if values["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    minimum_frames = temporal_frame_floor(conv_type)
    if values["frames"] < minimum_frames:
        raise ValueError(
            "frames must be at least {} for conv_type={}".format(
                minimum_frames,
                int(conv_type),
            )
        )
    if values["height"] < 32 or values["width"] < 32:
        raise ValueError("height and width must each be at least 32")
    if values["warmup"] < 0:
        raise ValueError("warmup must be non-negative")
    if values["iterations"] < 1:
        raise ValueError("iterations must be positive")
    return values


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "trainable_ratio": float(trainable) / float(total) if total else 0.0,
        "parameter_bytes_fp32": int(total * 4),
    }


def import_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def synthetic_gloss_dict(num_classes):
    if num_classes < 2:
        raise ValueError("num_classes must include blank and one gloss")
    return {
        "gloss{:04d}".format(class_id): (class_id,)
        for class_id in range(1, num_classes)
    }


def _load_config(path, root):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    parser = get_parser()
    with open(path, encoding="utf-8") as source:
        defaults = yaml.safe_load(source)
    parser.set_defaults(**defaults)
    args = parser.parse_args(["--config", str(path)])
    return path.resolve(), args


def _resolve_device(value):
    if value == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:{}".format(value))


def _move_inputs(device, batch_size, frames, height, width, num_classes):
    video = torch.randn(batch_size, frames, 3, height, width, device=device)
    video_lengths = torch.full(
        (batch_size,),
        frames,
        dtype=torch.long,
        device=device,
    )
    label = torch.ones(batch_size, dtype=torch.long, device=device)
    label_lengths = torch.ones(batch_size, dtype=torch.long, device=device)
    return video, video_lengths, label, label_lengths


def _forward(model, inputs, signer_ids=None):
    video, video_lengths, label, label_lengths = inputs
    kwargs = {
        "label": label,
        "label_lgt": label_lengths,
        "decode": False,
    }
    if getattr(model, "supports_signer", False):
        if signer_ids is None:
            signer_ids = torch.zeros(
                video.size(0),
                dtype=torch.long,
                device=video.device,
            )
        kwargs["signer_ids"] = signer_ids
    output = model(video, video_lengths, **kwargs)
    loss = model.criterion_calculation(
        output,
        label,
        label_lengths,
    )
    return output, loss


def _flop_count(callback):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    with FlopCounterMode(display=False) as counter:
        callback()
    return int(counter.get_total_flops())


def _time_calls(callback, warmup, iterations, device):
    for _ in range(int(warmup)):
        callback()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    durations = []
    for _ in range(int(iterations)):
        started = time.perf_counter()
        callback()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    return durations


def profile(
    config_path,
    device="cpu",
    batch_size=2,
    frames=16,
    height=224,
    width=224,
    warmup=2,
    iterations=5,
    root=ROOT,
):
    config_path, args = _load_config(config_path, Path(root).resolve())
    device = _resolve_device(str(device))
    shape = validate_profile_request(
        args.model_args.get("conv_type", 0),
        batch_size,
        frames,
        height,
        width,
        warmup,
        iterations,
    )
    batch_size = shape["batch_size"]
    frames = shape["frames"]
    height = shape["height"]
    width = shape["width"]
    warmup = shape["warmup"]
    iterations = shape["iterations"]
    model_class = import_class(args.model)
    gloss_dict = synthetic_gloss_dict(int(args.model_args["num_classes"]))
    model = model_class(
        **args.model_args,
        gloss_dict=gloss_dict,
        loss_weights=args.loss_weights,
    )
    model.to(device)
    model.train()
    set_training_epoch = getattr(model, "set_training_epoch", None)
    if set_training_epoch is not None:
        set_training_epoch(0)

    inputs = _move_inputs(
        device,
        int(batch_size),
        int(frames),
        int(height),
        int(width),
        int(args.model_args["num_classes"]),
    )
    signer_ids = torch.zeros(
        int(batch_size),
        dtype=torch.long,
        device=device,
    )

    def forward_call():
        with torch.set_grad_enabled(False):
            _forward(model, inputs, signer_ids)

    def train_step():
        model.zero_grad(set_to_none=True)
        _, loss = _forward(model, inputs, signer_ids)
        loss.backward()

    forward_flops = _flop_count(forward_call)
    train_step_flops = _flop_count(train_step)
    forward_times = _time_calls(
        forward_call,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    train_times = _time_calls(
        train_step,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = float(torch.cuda.max_memory_allocated(device))
        peak_reserved = float(torch.cuda.max_memory_reserved(device))
    result = {
        "config": str(config_path),
        "model": str(args.model),
        "device": str(device),
        "input_shape": [
            int(batch_size),
            int(frames),
            3,
            int(height),
            int(width),
        ],
        "loss_weights": {
            str(name): float(weight)
            for name, weight in args.loss_weights.items()
        },
        **parameter_counts(model),
        "forward_flops": forward_flops,
        "train_step_flops": train_step_flops,
        "forward_latency_ms_mean": statistics.mean(forward_times) * 1000.0,
        "forward_latency_ms_std": (
            statistics.pstdev(forward_times) * 1000.0
            if len(forward_times) > 1
            else 0.0
        ),
        "train_step_latency_ms_mean": statistics.mean(train_times) * 1000.0,
        "train_step_latency_ms_std": (
            statistics.pstdev(train_times) * 1000.0
            if len(train_times) > 1
            else 0.0
        ),
        "forward_samples_per_second": (
            float(batch_size) / statistics.mean(forward_times)
        ),
        "train_samples_per_second": (
            float(batch_size) / statistics.mean(train_times)
        ),
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "warmup_iterations": int(warmup),
        "measured_iterations": int(iterations),
        "measurement_note": (
            "FLOPs are operator counts from torch FlopCounterMode for the "
            "synthetic video branch; timings exclude data loading and "
            "decoding, and must be compared on the same GPU."
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = profile(
        args.config,
        device=args.device,
        batch_size=args.batch_size,
        frames=args.frames,
        height=args.height,
        width=args.width,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
