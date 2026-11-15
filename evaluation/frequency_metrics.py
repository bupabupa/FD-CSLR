import math
from collections import OrderedDict

import numpy as np
import torch


BIN_NAMES = ("head", "middle", "tail")


def align_occurrences(reference, hypothesis, num_classes=None):
    """Levenshtein alignment and class-wise occurrence TP/FP/FN counts."""
    reference = list(map(int, reference))
    hypothesis = list(map(int, hypothesis))
    if num_classes is None:
        values = reference + hypothesis
        num_classes = max(values) + 1 if values else 0
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distance = np.zeros((rows, cols), dtype=np.int32)
    distance[:, 0] = np.arange(rows)
    distance[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            distance[i, j] = min(
                distance[i - 1, j] + 1,
                distance[i, j - 1] + 1,
                distance[i - 1, j - 1] + substitution,
            )

    i = rows - 1
    j = cols - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitution = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if distance[i, j] == distance[i - 1, j - 1] + substitution:
                if substitution == 0:
                    tp[reference[i - 1]] += 1
                else:
                    fn[reference[i - 1]] += 1
                    fp[hypothesis[j - 1]] += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and distance[i, j] == distance[i - 1, j] + 1:
            fn[reference[i - 1]] += 1
            i -= 1
        else:
            fp[hypothesis[j - 1]] += 1
            j -= 1
    return {
        "edits": int(distance[-1, -1]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aggregate_alignment(references, hypotheses, num_classes):
    edits = 0
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for reference, hypothesis in zip(references, hypotheses):
        counts = align_occurrences(reference, hypothesis, num_classes)
        edits += counts["edits"]
        tp += counts["tp"]
        fp += counts["fp"]
        fn += counts["fn"]
    return {"edits": edits, "tp": tp, "fp": fp, "fn": fn}


def sequence_error_counts(reference, hypothesis, num_classes):
    """Return edits and reference length for one sentence-level alignment."""
    counts = align_occurrences(reference, hypothesis, num_classes)
    reference_length = int(counts["tp"].sum() + counts["fn"].sum())
    return counts["edits"], reference_length


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _scores(tp, fp, fn):
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _bounded_confusion_rate(tp, fp, fn):
    """Rate penalizing both false negatives and false positives."""
    tp = np.asarray(tp, dtype=np.int64)
    fp = np.asarray(fp, dtype=np.int64)
    fn = np.asarray(fn, dtype=np.int64)
    return np.divide(
        fp + fn,
        tp + fp + fn,
        out=np.zeros(len(tp), dtype=np.float64),
        where=(tp + fp + fn) > 0,
    )


def occurrence_metrics(tp, fp, fn):
    """Micro and macro occurrence metrics for class-wise counts."""
    tp = np.asarray(tp, dtype=np.int64)
    fp = np.asarray(fp, dtype=np.int64)
    fn = np.asarray(fn, dtype=np.int64)
    micro = _scores(tp.sum(), fp.sum(), fn.sum())
    supported = tp + fn > 0
    per_class_precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros(len(tp), dtype=np.float64),
        where=tp + fp > 0,
    )
    per_class_recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros(len(tp), dtype=np.float64),
        where=tp + fn > 0,
    )
    denominator = per_class_precision + per_class_recall
    per_class_f1 = np.divide(
        2 * per_class_precision * per_class_recall,
        denominator,
        out=np.zeros(len(tp), dtype=np.float64),
        where=denominator > 0,
    )
    per_class_uwer = np.divide(
        fp + fn,
        tp + fn,
        out=np.zeros(len(tp), dtype=np.float64),
        where=(tp + fn) > 0,
    )
    class_balanced_error = (
        float(per_class_uwer[supported].mean()) if supported.any() else 0.0
    )
    return {
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_precision": float(per_class_precision[supported].mean())
        if supported.any()
        else 0.0,
        "macro_recall": float(per_class_recall[supported].mean())
        if supported.any()
        else 0.0,
        "macro_f1": float(per_class_f1[supported].mean())
        if supported.any()
        else 0.0,
        # uWER treats every supported gloss class equally; it is not
        # occurrence-weighted sentence WER.
        "uwer": class_balanced_error,
        "class_balanced_error": class_balanced_error,
        "support": int((tp + fn).sum()),
        "predicted": int((tp + fp).sum()),
        "true_positive": int(tp.sum()),
    }


def frequency_bins(train_counts, fractions=(0.2, 0.3, 0.5)):
    """Split classes by descending training frequency into ordered bins."""
    counts = np.asarray(train_counts, dtype=np.float64)
    if np.any(counts < 0):
        raise ValueError("train_counts cannot be negative")
    if len(fractions) != 3 or any(float(value) < 0.0 for value in fractions):
        raise ValueError("frequency fractions require three non-negative values")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("frequency fractions must sum to one")
    order = np.lexsort((np.arange(len(counts)), -counts))
    total = len(order)
    head_end = max(1, int(total * fractions[0])) if total else 0
    middle_end = int(total * (fractions[0] + fractions[1]))
    if total > head_end:
        middle_end = max(head_end + 1, middle_end)
    middle_end = min(total, middle_end)
    return OrderedDict(
        [
            ("head", order[:head_end].tolist()),
            ("middle", order[head_end:middle_end].tolist()),
            ("tail", order[middle_end:].tolist()),
        ]
    )


def frequency_metrics(tp, fp, fn, train_counts, fractions=(0.2, 0.3, 0.5)):
    classes = frequency_bins(train_counts, fractions=fractions)
    tp = np.asarray(tp, dtype=np.int64)
    fp = np.asarray(fp, dtype=np.int64)
    fn = np.asarray(fn, dtype=np.int64)
    output = {"overall": occurrence_metrics(tp, fp, fn), "bins": {}}
    for name, class_ids in classes.items():
        class_ids = np.asarray(class_ids, dtype=np.int64)
        output["bins"][name] = occurrence_metrics(
            tp[class_ids], fp[class_ids], fn[class_ids]
        )
        output["bins"][name]["num_classes"] = int(len(class_ids))
        output["bins"][name]["train_occurrences"] = int(
            np.asarray(train_counts)[class_ids].sum()
        )
    return output


def class_frequency_report(
    references,
    hypotheses,
    train_counts,
    blank_id=None,
    fractions=(0.2, 0.3, 0.5),
):
    """Build occurrence metrics for semantic (non-blank) classes."""
    train_counts = np.asarray(train_counts, dtype=np.int64)
    if train_counts.ndim != 1:
        raise ValueError("train_counts must be one-dimensional")
    num_classes = len(train_counts)
    counts = aggregate_alignment(references, hypotheses, num_classes)
    if blank_id is None:
        semantic_ids = np.arange(num_classes)
    else:
        if not 0 <= blank_id < num_classes:
            raise ValueError("blank_id is outside the class range")
        semantic_ids = np.delete(np.arange(num_classes), blank_id)
    report = frequency_metrics(
        counts["tp"][semantic_ids],
        counts["fp"][semantic_ids],
        counts["fn"][semantic_ids],
        train_counts[semantic_ids],
        fractions=fractions,
    )
    report["utterance_mean_wer"] = utterance_mean_wer(
        references,
        hypotheses,
        num_classes,
    )
    tp = counts["tp"][semantic_ids]
    fp = counts["fp"][semantic_ids]
    fn = counts["fn"][semantic_ids]
    report["per_class"] = {
        "class_ids": semantic_ids.tolist(),
        "train_counts": train_counts[semantic_ids].tolist(),
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "support": (tp + fn).tolist(),
        "error_rate": _bounded_confusion_rate(tp, fp, fn).tolist(),
        "uwer_error_rate": np.divide(
            fp + fn,
            tp + fn,
            out=np.zeros(len(tp), dtype=np.float64),
            where=(tp + fn) > 0,
        ).tolist(),
        "precision": np.divide(
            tp,
            tp + fp,
            out=np.zeros(len(tp), dtype=np.float64),
            where=(tp + fp) > 0,
        ).tolist(),
        "recall": np.divide(
            tp,
            tp + fn,
            out=np.zeros(len(tp), dtype=np.float64),
            where=(tp + fn) > 0,
        ).tolist(),
    }
    precision = np.asarray(report["per_class"]["precision"])
    recall = np.asarray(report["per_class"]["recall"])
    denominator = precision + recall
    report["per_class"]["f1"] = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros(len(tp), dtype=np.float64),
        where=denominator > 0,
    ).tolist()
    return report


def utterance_mean_wer(references, hypotheses, num_classes):
    """Mean sentence-level WER with every utterance weighted equally."""
    values = []
    for reference, hypothesis in zip(references, hypotheses):
        edits, reference_length = sequence_error_counts(
            reference,
            hypothesis,
            num_classes,
        )
        values.append(_safe_ratio(edits, reference_length))
    return float(np.mean(values)) if values else 0.0


def gloss_occurrence_counts(samples, gloss_dict, num_classes=None):
    """Count training gloss occurrences directly from PHOENIX info entries."""
    if num_classes is None:
        ids = [int(value[0]) for value in gloss_dict.values()]
        num_classes = max(ids, default=0) + 1
    counts = np.zeros(num_classes, dtype=np.int64)
    if isinstance(samples, dict):
        samples = samples.values()
    for sample in samples:
        label = sample.get("label", "") if isinstance(sample, dict) else ""
        for gloss in label.split():
            value = gloss_dict.get(gloss)
            if value is None:
                continue
            class_id = int(value[0])
            if not 0 <= class_id < num_classes:
                raise ValueError("gloss dictionary contains an invalid class id")
            counts[class_id] += 1
    return counts


def _ece(confidence, correct, sample_weight=None, num_bins=15):
    confidence = confidence.reshape(-1)
    correct = correct.reshape(-1).to(confidence.dtype)
    if confidence.numel() == 0:
        return 0.0
    if sample_weight is None:
        sample_weight = torch.ones_like(confidence)
    else:
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
        error = error + weight * (accuracy - mean_confidence).abs()
    return float(error.item())


def frame_calibration_metrics(
    log_probs,
    targets,
    source_prior=None,
    target_prior=None,
    frame_mask=None,
    num_bins=15,
):
    """NLL, Brier, accuracy, ECE and prior-weighted ECE for frame scores."""
    log_probs = torch.as_tensor(log_probs)
    targets = torch.as_tensor(targets, dtype=torch.long, device=log_probs.device)
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [N,C]")
    if log_probs.shape[0] != targets.numel():
        raise ValueError("targets must have one value per frame")
    if frame_mask is not None:
        frame_mask = torch.as_tensor(
            frame_mask, dtype=torch.bool, device=log_probs.device
        )
        log_probs = log_probs[frame_mask]
        targets = targets[frame_mask]
    if targets.numel() == 0:
        raise ValueError("at least one frame is required")

    probs = log_probs.exp()
    confidence, predictions = probs.max(dim=-1)
    correct = predictions == targets
    nll = float(torch.nn.functional.nll_loss(log_probs, targets).item())
    one_hot = torch.nn.functional.one_hot(
        targets, num_classes=log_probs.shape[-1]
    ).to(probs.dtype)
    brier = float((probs - one_hot).square().sum(dim=-1).mean().item())
    metrics = {
        "nll": nll,
        "brier": brier,
        "accuracy": float(correct.to(probs.dtype).mean().item()),
        "ece": _ece(confidence, correct, num_bins=num_bins),
        "num_frames": int(targets.numel()),
    }
    if source_prior is not None and target_prior is not None:
        source_prior = torch.as_tensor(
            source_prior, dtype=probs.dtype, device=probs.device
        )
        target_prior = torch.as_tensor(
            target_prior, dtype=probs.dtype, device=probs.device
        )
        weights = target_prior[targets] / source_prior[targets].clamp_min(1e-12)
        metrics["prior_weighted_ece"] = _ece(
            confidence, correct, sample_weight=weights, num_bins=num_bins
        )
    return metrics
