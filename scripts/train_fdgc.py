import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils
from main import Processor
from modules.prototype import ctc_viterbi_path
from dataset.splits import load_split_manifest
from evaluation.slr_eval.wer_calculation import evaluate as evaluate_wer


def parse_args():
    parser = utils.get_parser()
    parser.add_argument(
        "--fdgc-use-difficulty",
        action="store_true",
        help="enable the FDGC-FD difficulty input; default is FDGC-F",
    )
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
    return args


def load_difficulty(path, num_nonblank):
    difficulty = torch.zeros(num_nonblank)
    if path is None:
        return difficulty
    with open(path, encoding="utf-8") as source:
        report = json.load(source)
    values = report.get("per_class")
    if values is None:
        raise ValueError("difficulty report is missing per_class metrics")
    for class_id, error_rate in zip(
        values["class_ids"], values["error_rate"]
    ):
        class_id = int(class_id)
        if 1 <= class_id <= num_nonblank:
            difficulty[class_id - 1] = float(error_rate)
    maximum = difficulty.max()
    if maximum > 0:
        difficulty /= maximum
    return difficulty


def configure_difficulty(path, num_nonblank, use_difficulty):
    if not use_difficulty:
        return torch.zeros(num_nonblank)
    if path is None:
        raise ValueError(
            "--difficulty-report is required with --fdgc-use-difficulty"
        )
    return load_difficulty(path, num_nonblank)


def difficulty_from_error_counts(error_counts, total_counts):
    error_counts = torch.as_tensor(error_counts, dtype=torch.float64)
    total_counts = torch.as_tensor(total_counts, dtype=torch.float64)
    if error_counts.shape != total_counts.shape:
        raise ValueError("difficulty counts must have matching shapes")
    if error_counts.ndim != 1:
        raise ValueError("difficulty counts must be one-dimensional")
    if (error_counts < 0).any() or (total_counts < 0).any():
        raise ValueError("difficulty counts must be non-negative")
    if (error_counts > total_counts).any():
        raise ValueError("class errors cannot exceed class frame counts")
    difficulty = error_counts / total_counts.clamp_min(1.0)
    maximum = difficulty.max() if difficulty.numel() else difficulty.new_tensor(0.0)
    if maximum > 0:
        difficulty = difficulty / maximum
    statistics = {
        "source": "D_cal_viterbi_error",
        "num_frames": int(total_counts.sum().item()),
        "num_covered_classes": int((total_counts > 0).sum().item()),
        "num_missing_classes": int((total_counts == 0).sum().item()),
        "max_raw_error_rate": float(maximum.item()),
    }
    return difficulty, statistics


@torch.no_grad()
def estimate_calibration_difficulty(model, loader, device, num_nonblank):
    error_counts = torch.zeros(num_nonblank, dtype=torch.float64)
    total_counts = torch.zeros(num_nonblank, dtype=torch.float64)
    model.eval()
    for data in tqdm(loader, desc="D_cal difficulty estimation"):
        video = device.data_to_device(data[0])
        video_length = device.data_to_device(data[1])
        labels = device.data_to_device(data[2])
        label_lengths = device.data_to_device(data[3])
        model_kwargs = {"decode": False}
        if getattr(model, "supports_signer", False):
            model_kwargs["signer_ids"] = device.data_to_device(data[4])
        output = model(video, video_length, **model_kwargs)
        logits = output["sequence_logits"].detach()
        targets, frame_mask = extract_alignment_targets(
            logits, labels, label_lengths, output["feat_len"]
        )
        target_frames = targets[frame_mask]
        predicted_frames = logits.argmax(dim=-1)[frame_mask]
        nonblank = target_frames != 0
        class_ids = (target_frames[nonblank] - 1).cpu()
        errors = predicted_frames[nonblank].ne(
            target_frames[nonblank]
        ).double().cpu()
        error_counts.index_add_(0, class_ids, errors)
        total_counts.index_add_(
            0, class_ids, torch.ones_like(errors)
        )
    return difficulty_from_error_counts(error_counts, total_counts)


def is_better_fdgc(
    dev_wer,
    calibration_nll,
    best_dev_wer,
    best_nll,
    wer_tolerance=1e-9,
):
    """Select by Dev WER, then by held-out calibration NLL on exact ties."""
    if dev_wer < best_dev_wer - wer_tolerance:
        return True
    return (
        abs(dev_wer - best_dev_wer) <= wer_tolerance
        and calibration_nll < best_nll - 1e-12
    )


def is_better_calibration(calibration_nll, calibration_ece,
                          best_nll, best_ece, tolerance=1e-12):
    """Select calibration state on D_cal alone, never on D_sel."""
    if calibration_nll < best_nll - tolerance:
        return True
    return (
        abs(calibration_nll - best_nll) <= tolerance
        and calibration_ece < best_ece - tolerance
    )


def extract_alignment_targets(logits, labels, label_lengths, feature_lengths):
    time_steps, batch_size, _ = logits.shape
    blank_id = 0
    targets = torch.full(
        (time_steps, batch_size),
        blank_id,
        dtype=torch.long,
        device=logits.device,
    )
    frame_mask = torch.zeros(
        (time_steps, batch_size),
        dtype=torch.bool,
        device=logits.device,
    )
    log_probs = logits.detach().log_softmax(dim=-1)
    offset = 0
    lengths = label_lengths.detach().cpu().tolist()
    feature_lengths = feature_lengths.detach().cpu().tolist()
    for batch_idx, label_length in enumerate(lengths):
        label_length = int(label_length)
        sequence_length = min(
            int(feature_lengths[batch_idx]), time_steps
        )
        target = labels[offset:offset + label_length]
        offset += label_length
        if sequence_length <= 0:
            continue
        try:
            path = ctc_viterbi_path(
                log_probs[:sequence_length, batch_idx],
                target,
                blank_id=blank_id,
            )
        except ValueError:
            # Keep impossible targets out of calibration and difficulty
            # accounting instead of poisoning the batch.
            continue
        targets[:sequence_length, batch_idx] = path
        frame_mask[:sequence_length, batch_idx] = True
    return targets, frame_mask


def ctc_calibration_loss(
    output,
    labels,
    label_lengths,
    feature_lengths,
    blank_id=0,
    regularization=1e-3,
):
    """Maximize the standard CTC sequence likelihood after calibration."""
    log_probs = output["log_probs"]
    valid_losses = []
    label_offset = 0
    for sample_idx, target_length in enumerate(
        label_lengths.cpu().int().tolist()
    ):
        target = labels[
            label_offset:label_offset + target_length
        ].cpu().int()
        label_offset += target_length
        sample_loss = F.ctc_loss(
            log_probs[:, sample_idx:sample_idx + 1],
            target,
            feature_lengths[sample_idx:sample_idx + 1].cpu().int(),
            label_lengths[sample_idx:sample_idx + 1].cpu().int(),
            blank=blank_id,
            reduction="none",
            zero_infinity=False,
        ).reshape(())
        if torch.isfinite(sample_loss):
            valid_losses.append(sample_loss)
    if not valid_losses:
        raise FloatingPointError(
            "FDGC CTC calibration produced no finite sample losses"
        )
    sequence_loss = torch.stack(valid_losses).mean()
    time_steps = output["log_probs"].size(0)
    valid_mask = (
        torch.arange(time_steps, device=output["log_probs"].device).view(-1, 1)
        < feature_lengths.to(output["log_probs"].device)
        .clamp(0, time_steps)
        .view(1, -1)
    )
    tau = output["tau"][valid_mask]
    if tau.numel() == 0:
        return sequence_loss
    return sequence_loss + float(regularization) * tau.square().mean()


def _ece(confidence, correct, sample_weight, num_bins=15):
    confidence = confidence.reshape(-1)
    correct = correct.reshape(-1).to(confidence.dtype)
    sample_weight = sample_weight.reshape(-1).to(confidence.dtype)
    sample_weight = sample_weight / sample_weight.sum().clamp_min(1e-12)
    bin_ids = (confidence * num_bins).long().clamp(0, num_bins - 1)
    error = confidence.new_tensor(0.0)
    for bin_id in range(num_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        weight = sample_weight[mask].sum()
        accuracy = (correct[mask] * sample_weight[mask]).sum() / weight
        mean_confidence = (
            confidence[mask] * sample_weight[mask]
        ).sum() / weight
        error += weight * (accuracy - mean_confidence).abs()
    return error


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, reference_token in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + int(reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def _write_ctm(path, sample_ids, sentences):
    with open(path, "w", encoding="utf-8") as output:
        for sample_id, sentence in zip(sample_ids, sentences):
            for word_index, token in enumerate(sentence):
                word = token[0] if isinstance(token, (tuple, list)) else token
                output.write(
                    "{} 1 {:.2f} {:.2f} {}\n".format(
                        sample_id,
                        word_index / 100.0,
                        (word_index + 1) / 100.0,
                        word,
                    )
                )


@torch.no_grad()
def evaluate_dev_wer(model, loader, device, dataset_info=None):
    model.eval()
    gloss_to_id = {
        str(gloss): int(value[0])
        for gloss, value in loader.dataset.dict.items()
    }
    total_edits = 0
    total_reference_length = 0
    total_info = []
    total_sentences = []
    for data in tqdm(loader, desc="FDGC Dev selection"):
        video = device.data_to_device(data[0])
        video_length = device.data_to_device(data[1])
        labels = device.data_to_device(data[2])
        label_lengths = device.data_to_device(data[3])
        model_kwargs = {"decode": True}
        if getattr(model, "supports_signer", False):
            model_kwargs["signer_ids"] = device.data_to_device(data[4])
        hypotheses = model(video, video_length, **model_kwargs)[
            "recognized_sents"
        ]
        total_info.extend(
            file_name.split("|")[0] for file_name in data[-1]
        )
        total_sentences.extend(hypotheses)
        flat_labels = labels.detach().cpu().tolist()
        offset = 0
        for batch_idx, label_length in enumerate(
            label_lengths.detach().cpu().tolist()
        ):
            label_length = int(label_length)
            reference = flat_labels[offset:offset + label_length]
            offset += label_length
            hypothesis = []
            for token in hypotheses[batch_idx]:
                gloss = token[0] if isinstance(token, (tuple, list)) else token
                class_id = gloss_to_id.get(str(gloss))
                if class_id is None:
                    hypothesis.append(-1)
                else:
                    hypothesis.append(class_id)
            total_edits += _edit_distance(reference, hypothesis)
            total_reference_length += len(reference)
    token_wer = 100.0 * total_edits / max(total_reference_length, 1)
    if dataset_info is None:
        return token_wer
    with tempfile.TemporaryDirectory(prefix="fdgc_dev_selection_") as temp:
        prefix = Path(temp) / ""
        output_file = "output-hypothesis-dev.ctm"
        _write_ctm(prefix / output_file, total_info, total_sentences)
        return evaluate_wer(
            prefix=str(prefix),
            mode="dev",
            evaluate_dir=dataset_info["evaluation_dir"],
            evaluate_prefix=dataset_info["evaluation_prefix"],
            output_file=output_file,
        )


@torch.no_grad()
def evaluate_calibration(model, loader, device, difficulty, source_prior):
    confidence_values = []
    correct_values = []
    target_probability_values = []
    probability_square_values = []
    target_values = []
    prior_weights = []
    model.eval()
    for data in tqdm(loader, desc="FDGC calibration"):
        video = device.data_to_device(data[0])
        video_length = device.data_to_device(data[1])
        labels = device.data_to_device(data[2])
        label_lengths = device.data_to_device(data[3])
        model_kwargs = {"decode": False}
        if getattr(model, "supports_signer", False):
            model_kwargs["signer_ids"] = device.data_to_device(data[4])
        output = model(video, video_length, **model_kwargs)
        logits = output["sequence_logits"].detach()
        targets, frame_mask = extract_alignment_targets(
            logits, labels, label_lengths, output["feat_len"]
        )
        fdgc_output = model.fdgc(
            logits,
            source_prior=source_prior,
            difficulty=difficulty,
            frame_mask=frame_mask,
            input_is_log_probs=(
                output.get("sequence_log_probs") is not None
            ),
        )
        log_probs = fdgc_output["log_probs"][frame_mask]
        targets = targets[frame_mask]
        probabilities = log_probs.exp()
        confidence, predictions = probabilities.max(dim=-1)
        target_probability = probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        class_weights = torch.ones_like(confidence)
        nonblank = targets != 0
        class_weights[nonblank] = (
            1.0 / source_prior[targets[nonblank] - 1]
        )
        confidence_values.append(confidence.cpu())
        correct_values.append((predictions == targets).cpu())
        target_probability_values.append(target_probability.cpu())
        probability_square_values.append(
            probabilities.square().sum(dim=-1).cpu()
        )
        target_values.append(targets.cpu())
        prior_weights.append(class_weights.cpu())

    confidence = torch.cat(confidence_values)
    correct = torch.cat(correct_values)
    target_probability = torch.cat(target_probability_values)
    probability_square = torch.cat(probability_square_values)
    targets = torch.cat(target_values)
    prior_weights = torch.cat(prior_weights)
    ece = _ece(confidence, correct, torch.ones_like(confidence))
    prior_ece = _ece(confidence, correct, prior_weights)
    return {
        "nll": float(-target_probability.clamp_min(1e-12).log().mean()),
        "brier": float(
            (1.0 - 2.0 * target_probability + probability_square).mean()
        ),
        "accuracy": float(correct.float().mean()),
        "ece": float(ece),
        "prior_weighted_ece": float(prior_ece),
        "num_frames": int(targets.numel()),
        "blank_fraction": float((targets == 0).float().mean()),
    }


def main():
    args = parse_args()
    processor = Processor(args)
    model = processor.model
    device = processor.device
    loader = processor.data_loader["cal"]
    if model.fdgc is None:
        raise ValueError("configured model does not have FDGC enabled")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fdgc.parameters():
        parameter.requires_grad_(True)
    if args.fdgc_use_difficulty:
        difficulty, difficulty_statistics = estimate_calibration_difficulty(
            model,
            loader,
            device,
            model.fdgc_difficulty.numel(),
        )
        difficulty_source = "D_cal_viterbi_error"
    else:
        difficulty = torch.zeros(model.fdgc_difficulty.numel())
        difficulty_statistics = {
            "source": "none",
            "num_frames": 0,
            "num_covered_classes": 0,
            "num_missing_classes": int(model.fdgc_difficulty.numel()),
            "max_raw_error_rate": 0.0,
        }
        difficulty_source = "none"
    difficulty = difficulty.to(model.fdgc_difficulty.device)
    model.set_fdgc_difficulty(difficulty)
    source_prior = model._nonblank_source_prior().detach()

    model.eval()
    identity_metrics = evaluate_calibration(
        model, loader, device, difficulty, source_prior
    )
    identity_dev_wer = evaluate_dev_wer(
        model,
        processor.data_loader["dev"],
        device,
        dataset_info=args.dataset_info,
    )
    optimizer = torch.optim.Adam(
        model.fdgc.parameters(),
        lr=args.fdgc_lr,
        weight_decay=args.fdgc_weight_decay,
    )
    best_loss = float(identity_metrics["nll"])
    best_ece = float(identity_metrics["ece"])
    best_epoch = 0
    best_state = copy.deepcopy(model.fdgc.state_dict())
    history = [
        {
            "epoch": 0,
            "loss": float(identity_metrics["nll"]),
            **identity_metrics,
        }
    ]
    for epoch in range(args.fdgc_epochs):
        model.eval()
        losses = []
        for data in tqdm(
            loader, desc=f"FDGC epoch {epoch + 1}/{args.fdgc_epochs}"
        ):
            video = device.data_to_device(data[0])
            video_length = device.data_to_device(data[1])
            labels = device.data_to_device(data[2])
            label_lengths = device.data_to_device(data[3])
            model_kwargs = {"decode": False}
            if getattr(model, "supports_signer", False):
                model_kwargs["signer_ids"] = device.data_to_device(data[4])
            with torch.no_grad():
                output = model(video, video_length, **model_kwargs)
                if args.fdgc_objective == "viterbi":
                    targets, frame_mask = extract_alignment_targets(
                        output["sequence_logits"],
                        labels,
                        label_lengths,
                        output["feat_len"],
                    )
                else:
                    targets = None
                    frame_mask = None
            if args.fdgc_objective == "viterbi":
                fdgc_output = model.fdgc(
                    output["sequence_logits"].detach(),
                    source_prior=source_prior,
                    difficulty=difficulty,
                    frame_mask=frame_mask,
                    input_is_log_probs=(
                        output.get("sequence_log_probs") is not None
                    ),
                )
                loss = model.fdgc.calibration_loss(
                    fdgc_output, targets, frame_mask
                )
            else:
                fdgc_output = model.fdgc(
                    output["sequence_logits"].detach(),
                    source_prior=source_prior,
                    difficulty=difficulty,
                    input_is_log_probs=(
                        output.get("sequence_log_probs") is not None
                    ),
                )
                loss = ctc_calibration_loss(
                    fdgc_output,
                    labels,
                    label_lengths,
                    output["feat_len"],
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite FDGC loss at epoch {}, batch {}".format(
                        epoch + 1,
                        len(losses),
                    )
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.fdgc.parameters(), max_norm=5.0
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(len(losses), 1)
        metrics = evaluate_calibration(
            model, loader, device, difficulty, source_prior
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": mean_loss,
                **metrics,
            }
        )
        print(json.dumps(history[-1], sort_keys=True))
        calibration_nll = float(metrics["nll"])
        calibration_ece = float(metrics["ece"])
        if is_better_calibration(
            calibration_nll,
            calibration_ece,
            best_loss,
            best_ece,
        ):
            best_loss = calibration_nll
            best_ece = calibration_ece
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.fdgc.state_dict())

    model.fdgc.load_state_dict(best_state)
    selected_dev_wer = evaluate_dev_wer(
        model,
        processor.data_loader["dev"],
        device,
        dataset_info=args.dataset_info,
    )
    manifest = load_split_manifest(
        args.split_manifest, os.path.dirname(args.split_manifest)
    )
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_fdgc.pt"
    torch.save(
        {
            "fdgc_state_dict": best_state,
            "fdgc_difficulty": difficulty.detach().cpu(),
            "split_sha256": manifest["split_sha256"],
            "source_prior": source_prior.detach().cpu(),
            "identity_metrics": identity_metrics,
            "identity_dev_wer": identity_dev_wer,
            "best_loss": best_loss,
            "best_dev_wer": selected_dev_wer,
            "best_epoch": best_epoch,
            "objective": args.fdgc_objective,
            "use_difficulty": bool(args.fdgc_use_difficulty),
            "difficulty_source": difficulty_source,
            "difficulty_statistics": difficulty_statistics,
            "selection_rule": (
                "minimum D_cal NLL; D_cal ECE tie-break; "
                "D_sel is evaluated once after selection; D_test is never loaded"
            ),
            "history": history,
            "difficulty_report": args.difficulty_report,
            "config": vars(args),
        },
        checkpoint_path,
    )
    with open(
        output_dir / "fdgc_metrics.json", "w", encoding="utf-8"
    ) as output:
        json.dump(
            {
                "identity": identity_metrics,
                "identity_dev_wer": identity_dev_wer,
                "history": history,
                "best_loss": best_loss,
                "best_dev_wer": selected_dev_wer,
                "best_epoch": best_epoch,
                "objective": args.fdgc_objective,
                "use_difficulty": bool(args.fdgc_use_difficulty),
                "difficulty_source": difficulty_source,
                "difficulty_statistics": difficulty_statistics,
                "selection_rule":
                    "minimum D_cal NLL; D_cal ECE tie-break; "
                    "D_sel is evaluated once after selection; "
                    "D_test is never loaded",
                "split_sha256": manifest["split_sha256"],
            },
            output,
            indent=2,
            sort_keys=True,
        )
    print(f"saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
