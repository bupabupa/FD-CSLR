import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils
from dataset.splits import load_split_manifest
from evaluation.slr_eval.wer_calculation import evaluate as evaluate_wer
from main import Processor
from modules.calibration import (
    FUSION_PARAMETERIZATION_VERSION,
    MultiChannelCalibrator,
)
from scripts.train_fdgc import (
    _ece,
    _write_ctm,
    evaluate_dev_wer,
    extract_alignment_targets,
    is_better_calibration,
)


def parse_args():
    parser = utils.get_parser()
    parser.add_argument("--calibration-epochs", type=int, default=10)
    parser.add_argument("--calibration-lr", type=float, default=0.01)
    parser.add_argument("--calibration-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--calibration-regularization", type=float, default=1e-4
    )
    parser.add_argument("--min-temperature", type=float, default=0.25)
    parser.add_argument("--max-temperature", type=float, default=4.0)
    initial = parser.parse_args()
    if initial.config is None:
        raise ValueError("--config is required")
    with open(initial.config, encoding="utf-8") as source:
        defaults = yaml.load(source, Loader=yaml.FullLoader)
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    with open(
        f"./configs/{args.dataset}.yaml", encoding="utf-8"
    ) as source:
        args.dataset_info = yaml.load(source, Loader=yaml.FullLoader)
    if args.load_weights is None:
        raise ValueError("--load-weights is required")
    if args.split_manifest is None:
        raise ValueError("--split-manifest is required")
    args.phase = "test"
    args.load_splits = ["cal", "dev"]
    # The base checkpoint predates the calibrator state. It is merged only
    # after fitting, so the model must be constructed without those keys here.
    args.model_args = dict(args.model_args)
    args.model_args["multichannel_enabled"] = False
    return args


def channel_output(model, calibrator, data, device, decode=False):
    video = device.data_to_device(data[0])
    video_length = device.data_to_device(data[1])
    labels = device.data_to_device(data[2])
    label_lengths = device.data_to_device(data[3])
    model_kwargs = {"decode": decode}
    if getattr(model, "supports_signer", False):
        model_kwargs["signer_ids"] = device.data_to_device(data[4])
    output = model(video, video_length, **model_kwargs)
    if output.get("text_logits") is None:
        raise ValueError(
            "multichannel calibration requires text prototype logits"
        )
    calibrated = calibrator(
        output["base_logits"],
        output["prototype_logits"],
        output["text_logits"],
    )
    return (
        output,
        calibrated,
        labels,
        label_lengths,
        video_length,
    )


@torch.no_grad()
def evaluate_base_calibration(model, loader, device):
    confidence_values = []
    correct_values = []
    target_probability_values = []
    probability_square_values = []
    target_values = []
    model.eval()
    for data in tqdm(loader, desc="Base D_cal metrics"):
        output, _, labels, label_lengths, _ = channel_output(
            model, _IdentityCalibrator(), data, device, decode=False
        )
        logits = output["sequence_logits"].detach()
        targets, frame_mask = extract_alignment_targets(
            logits, labels, label_lengths, output["feat_len"]
        )
        log_probs = logits.log_softmax(dim=-1)[frame_mask]
        targets = targets[frame_mask]
        probabilities = log_probs.exp()
        confidence, predictions = probabilities.max(dim=-1)
        target_probability = probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        confidence_values.append(confidence.cpu())
        correct_values.append((predictions == targets).cpu())
        target_probability_values.append(target_probability.cpu())
        probability_square_values.append(
            probabilities.square().sum(dim=-1).cpu()
        )
        target_values.append(targets.cpu())
    return _metrics_from_values(
        confidence_values,
        correct_values,
        target_probability_values,
        probability_square_values,
        target_values,
    )


@torch.no_grad()
def evaluate_channel_calibration(model, calibrator, loader, device):
    confidence_values = []
    correct_values = []
    target_probability_values = []
    probability_square_values = []
    target_values = []
    model.eval()
    calibrator.eval()
    for data in tqdm(loader, desc="Calibrated D_cal metrics"):
        output, calibrated, labels, label_lengths, _ = channel_output(
            model, calibrator, data, device, decode=False
        )
        targets, frame_mask = extract_alignment_targets(
            output["sequence_logits"].detach(),
            labels,
            label_lengths,
            output["feat_len"],
        )
        log_probs = calibrated["log_probs"][frame_mask]
        targets = targets[frame_mask]
        probabilities = log_probs.exp()
        confidence, predictions = probabilities.max(dim=-1)
        target_probability = probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        confidence_values.append(confidence.cpu())
        correct_values.append((predictions == targets).cpu())
        target_probability_values.append(target_probability.cpu())
        probability_square_values.append(
            probabilities.square().sum(dim=-1).cpu()
        )
        target_values.append(targets.cpu())
    return _metrics_from_values(
        confidence_values,
        correct_values,
        target_probability_values,
        probability_square_values,
        target_values,
    )


def _metrics_from_values(
    confidence_values,
    correct_values,
    target_probability_values,
    probability_square_values,
    target_values,
):
    confidence = torch.cat(confidence_values)
    correct = torch.cat(correct_values)
    target_probability = torch.cat(target_probability_values)
    probability_square = torch.cat(probability_square_values)
    targets = torch.cat(target_values)
    return {
        "nll": float(-target_probability.clamp_min(1e-12).log().mean()),
        "brier": float(
            (1.0 - 2.0 * target_probability + probability_square).mean()
        ),
        "accuracy": float(correct.float().mean()),
        "ece": float(_ece(confidence, correct, torch.ones_like(confidence))),
        "num_frames": int(targets.numel()),
        "blank_fraction": float((targets == 0).float().mean()),
    }


@torch.no_grad()
def evaluate_calibrated_dev_wer(
    model,
    calibrator,
    loader,
    device,
    dataset_info,
):
    model.eval()
    calibrator.eval()
    total_info = []
    total_sentences = []
    for data in tqdm(loader, desc="Calibrated Dev selection"):
        output, calibrated, _, _, _ = channel_output(
            model, calibrator, data, device, decode=False
        )
        hypotheses = model.decoder.decode(
            calibrated["probabilities"],
            output["feat_len"],
            batch_first=False,
            probs=True,
        )
        total_info.extend(
            file_name.split("|")[0] for file_name in data[-1]
        )
        total_sentences.extend(hypotheses)
    with tempfile.TemporaryDirectory(
        prefix="multichannel_dev_selection_"
    ) as temporary:
        prefix = Path(temporary) / ""
        output_file = "output-hypothesis-dev.ctm"
        _write_ctm(
            prefix / output_file,
            total_info,
            total_sentences,
        )
        return evaluate_wer(
            prefix=str(prefix),
            mode="dev",
            evaluate_dir=dataset_info["evaluation_dir"],
            evaluate_prefix=dataset_info["evaluation_prefix"],
            output_file=output_file,
        )


class _IdentityCalibrator:
    """Expose the model's native fused output through the shared evaluator."""

    def eval(self):
        return self

    def __call__(self, linear_logits, visual_logits, text_logits):
        return {
            "probabilities": linear_logits.log_softmax(dim=-1).exp(),
            "log_probs": linear_logits.log_softmax(dim=-1),
        }


def main():
    args = parse_args()
    if args.calibration_epochs < 0:
        raise ValueError("--calibration-epochs must be non-negative")
    processor = Processor(args)
    model = processor.model
    device = processor.device
    calibration_loader = processor.data_loader["cal"]
    selection_loader = processor.data_loader["dev"]
    if model.text_prototype_head is None:
        raise ValueError("configured model does not have a text channel")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    calibrator = MultiChannelCalibrator(
        num_classes=model.classifier.out_features,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    ).to(device.output_device)
    initial_state = copy.deepcopy(calibrator.state_dict())

    identity_metrics = evaluate_base_calibration(
        model, calibration_loader, device
    )
    observed_metrics = evaluate_channel_calibration(
        model, calibrator, calibration_loader, device
    )
    best_loss = float(identity_metrics["nll"])
    best_ece = float(identity_metrics["ece"])
    best_epoch = -1
    best_channel = "identity"
    best_state = copy.deepcopy(initial_state)
    history = [{
        "epoch": 0,
        "stage": "identity",
        "nll": float(identity_metrics["nll"]),
        **identity_metrics,
        "initial_mixture": {
            "nll": float(observed_metrics["nll"]),
            "ece": float(observed_metrics["ece"]),
        },
    }]
    print(json.dumps(history[-1], sort_keys=True))

    optimizer = torch.optim.Adam(
        calibrator.parameters(),
        lr=args.calibration_lr,
        weight_decay=args.calibration_weight_decay,
    )
    for epoch in range(args.calibration_epochs):
        calibrator.train()
        model.eval()
        losses = []
        for data in tqdm(
            calibration_loader,
            desc=(
                f"Calibration epoch {epoch + 1}/"
                f"{args.calibration_epochs}"
            ),
        ):
            with torch.no_grad():
                output, _, labels, label_lengths, _ = channel_output(
                    model,
                    calibrator,
                    data,
                    device,
                    decode=False,
                )
                targets, frame_mask = extract_alignment_targets(
                    output["sequence_logits"].detach(),
                    labels,
                    label_lengths,
                    output["feat_len"],
                )
            calibrated = calibrator(
                output["base_logits"].detach(),
                output["prototype_logits"].detach(),
                output["text_logits"].detach(),
            )
            loss = calibrator.calibration_loss(
                calibrated,
                targets,
                frame_mask,
                regularization=args.calibration_regularization,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite multichannel loss at epoch {}, batch {}".format(
                        epoch + 1,
                        len(losses),
                    )
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                calibrator.parameters(), max_norm=5.0
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_channel_calibration(
            model, calibrator, calibration_loader, device
        )
        record = {
            "epoch": epoch + 1,
            "stage": "calibrated",
            "loss": sum(losses) / max(len(losses), 1),
            **metrics,
            "temperatures": (
                calibrator.temperatures().detach().cpu().tolist()
            ),
            "weights": calibrator.weights().detach().cpu().tolist(),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if is_better_calibration(
            float(metrics["nll"]),
            float(metrics["ece"]),
            best_loss,
            best_ece,
        ):
            best_loss = float(metrics["nll"])
            best_ece = float(metrics["ece"])
            best_epoch = epoch + 1
            best_channel = "calibrated"
            best_state = copy.deepcopy(calibrator.state_dict())

    identity_dev_wer = evaluate_dev_wer(
        model,
        selection_loader,
        device,
        dataset_info=args.dataset_info,
    )
    calibrator.load_state_dict(best_state)
    selected_dev_wer = (
        identity_dev_wer
        if best_channel == "identity"
        else evaluate_calibrated_dev_wer(
            model,
            calibrator,
            selection_loader,
            device,
            args.dataset_info,
        )
    )

    manifest = load_split_manifest(
        args.split_manifest, os.path.dirname(args.split_manifest)
    )
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_calibration.pt"
    torch.save(
        {
            "multichannel_state_dict": best_state,
            "fusion_parameterization_version": (
                FUSION_PARAMETERIZATION_VERSION
            ),
            "split_sha256": manifest["split_sha256"],
            "num_classes": int(model.classifier.out_features),
            "num_channels": 3,
            "channel_names": ["linear", "visual", "text"],
            "identity_metrics": identity_metrics,
            "identity_dev_wer": identity_dev_wer,
            "best_loss": best_loss,
            "best_dev_wer": selected_dev_wer,
            "best_epoch": best_epoch,
            "selected_channel": best_channel,
            "use_calibration": best_channel == "calibrated",
            "selection_rule": (
                "minimum D_cal NLL; D_cal ECE tie-break; "
                "D_sel is evaluated once after selection; D_test is never loaded"
            ),
            "history": history,
            "config": vars(args),
        },
        checkpoint_path,
    )
    with open(
        output_dir / "multichannel_metrics.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(
            {
                "identity": identity_metrics,
                "identity_dev_wer": identity_dev_wer,
                "history": history,
                "best_loss": best_loss,
                "best_dev_wer": selected_dev_wer,
                "best_epoch": best_epoch,
                "selected_channel": best_channel,
                "use_calibration": best_channel == "calibrated",
                "selection_rule": (
                    "minimum D_cal NLL; D_cal ECE tie-break; "
                    "D_sel is evaluated once after selection; "
                    "D_test is never loaded"
                ),
                "split_sha256": manifest["split_sha256"],
            },
            output,
            indent=2,
            sort_keys=True,
        )
    print(f"saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
