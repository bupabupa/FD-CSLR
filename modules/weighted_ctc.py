import math

import torch
import torch.nn.functional as F


def _frequency_signal(counts):
    counts = torch.as_tensor(counts, dtype=torch.float64).reshape(-1)
    if counts.numel() == 0:
        return counts.to(torch.float32)
    if not torch.isfinite(counts).all() or torch.any(counts < 0):
        raise ValueError("class counts must be finite and non-negative")
    values = torch.log1p(counts)
    minimum = values.min()
    maximum = values.max()
    if float((maximum - minimum).abs()) <= 1e-12:
        return torch.zeros_like(values, dtype=torch.float32)
    return ((maximum - values) / (maximum - minimum)).to(torch.float32)


def frequency_weights(
    counts,
    max_weight=2.0,
    temperature=1.0,
    mapping="linear",
):
    """Map lower-frequency classes to larger bounded weights.

    The linear mapping uses the exact review recommendation
    ``1 + (max_weight - 1) * s``. The sigmoid mapping is endpoint-normalized
    so that the observed head and tail still receive ``1`` and ``max_weight``.
    """
    max_weight = float(max_weight)
    temperature = float(temperature)
    if not math.isfinite(max_weight) or max_weight < 1.0:
        raise ValueError("max_weight must be finite and at least one")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    signal = _frequency_signal(counts)
    if mapping == "linear":
        normalized = signal
    elif mapping == "sigmoid":
        scaled = signal / temperature
        low = torch.sigmoid(torch.zeros((), dtype=scaled.dtype))
        high = torch.sigmoid(torch.tensor(1.0 / temperature, dtype=scaled.dtype))
        normalized = (torch.sigmoid(scaled) - low) / (high - low).clamp_min(1e-12)
    else:
        raise ValueError("mapping must be linear or sigmoid")
    weights = 1.0 + (max_weight - 1.0) * normalized.clamp(0.0, 1.0)
    return weights.to(torch.float32)


def class_weights_from_counts(
    counts,
    blank_id=0,
    blank_weight=1.0,
    max_weight=2.0,
    temperature=1.0,
    mapping="linear",
):
    """Build a full ``C+1`` weight vector from non-blank class counts."""
    counts = torch.as_tensor(counts, dtype=torch.float64).reshape(-1)
    blank_weight = float(blank_weight)
    if not math.isfinite(blank_weight) or blank_weight <= 0.0:
        raise ValueError("blank_weight must be finite and positive")
    nonblank = frequency_weights(
        counts,
        max_weight=max_weight,
        temperature=temperature,
        mapping=mapping,
    )
    output = counts.new_full((len(nonblank) + 1,), blank_weight).float()
    nonblank_ids = torch.arange(len(nonblank) + 1) != int(blank_id)
    output[nonblank_ids] = nonblank
    return output


def class_weights_from_prior(
    prior,
    blank_id=0,
    blank_weight=1.0,
    max_weight=2.0,
    temperature=1.0,
    mapping="linear",
):
    prior = torch.as_tensor(prior, dtype=torch.float64).reshape(-1)
    if not torch.isfinite(prior).all() or torch.any(prior < 0):
        raise ValueError("class prior must be finite and non-negative")
    if int(blank_id) < 0 or int(blank_id) >= prior.numel():
        raise ValueError("blank_id is outside the class prior")
    nonblank = torch.cat([prior[:blank_id], prior[blank_id + 1:]])
    return class_weights_from_counts(
        nonblank,
        blank_id=blank_id,
        blank_weight=blank_weight,
        max_weight=max_weight,
        temperature=temperature,
        mapping=mapping,
    )


def weighted_emission_log_probs(logits, class_weights):
    """Create a normalized weighted distribution before CTC marginalization."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [T,B,C]")
    weights = torch.as_tensor(
        class_weights,
        dtype=logits.dtype,
        device=logits.device,
    ).reshape(-1)
    if weights.numel() != logits.size(-1):
        raise ValueError("class weights must match the emission classes")
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("class weights must be finite and positive")
    return F.log_softmax(logits + weights.log().view(1, 1, -1), dim=-1)


def class_weighted_ctc_loss(
    logits,
    targets,
    input_lengths,
    target_lengths,
    class_weights,
    blank=0,
):
    log_probs = weighted_emission_log_probs(logits, class_weights)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=False,
    )
