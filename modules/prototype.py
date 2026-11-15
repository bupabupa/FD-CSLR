import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extend_targets(target, blank_id=0):
    length = target.numel()
    extended = target.new_full((length * 2 + 1,), blank_id)
    if length:
        extended[1::2] = target
    return extended


def _can_skip(extended, state, blank_id):
    return (
        state % 2 == 1
        and state + 2 < extended.numel()
        and extended[state] != blank_id
        and extended[state] != extended[state + 2]
    )


def ctc_posterior(log_probs, target, blank_id=0):
    """Return per-frame CTC class occupancy for one unsegmented sequence."""
    if target.numel() == 0:
        posterior = torch.zeros(
            (log_probs.size(0), log_probs.size(1)),
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        posterior[:, blank_id] = 1.0
        return posterior

    time_steps, num_classes = log_probs.shape
    extended = _extend_targets(target.to(log_probs.device), blank_id=blank_id)
    states = extended.numel()
    if target.numel() > time_steps:
        raise ValueError(
            f"CTC target has {target.numel()} symbols but only "
            f"{time_steps} frames are available"
        )
    repeated = (target[1:] == target[:-1]).sum().item()
    if target.numel() + repeated > time_steps:
        raise ValueError("CTC target cannot fit in the available frame count")

    neg_inf = log_probs.new_tensor(-1e30)
    skip_mask = torch.tensor(
        [
            bool(_can_skip(extended, state, blank_id))
            for state in range(states)
        ],
        dtype=torch.bool,
        device=log_probs.device,
    )
    alpha = log_probs.new_full((time_steps, states), -1e30)
    alpha[0, 0] = log_probs[0, blank_id]
    if states > 1:
        alpha[0, 1] = log_probs[0, extended[1]]

    for step in range(1, time_steps):
        previous = alpha[step - 1]
        candidates = [previous]
        candidates.append(F.pad(previous[:-1], (1, 0), value=neg_inf))
        skip = F.pad(previous[:-2], (2, 0), value=neg_inf)
        skip_source_mask = F.pad(skip_mask[:-2], (2, 0), value=False)
        skip = torch.where(skip_source_mask, skip, neg_inf)
        candidates.append(skip)
        alpha[step] = torch.logsumexp(torch.stack(candidates, dim=0), dim=0)
        alpha[step] = alpha[step] + log_probs[step, extended]

    beta = log_probs.new_full((time_steps, states), -1e30)
    beta[-1, states - 1] = log_probs.new_tensor(0.0)
    if states > 1 and extended[-2] != blank_id:
        beta[-1, states - 2] = log_probs.new_tensor(0.0)

    state_emissions = log_probs[:, extended]
    for step in range(time_steps - 2, -1, -1):
        following = beta[step + 1]
        next_emissions = state_emissions[step + 1]
        candidates = [following + next_emissions]
        advance = F.pad(following[1:], (0, 1), value=neg_inf)
        advance = advance + F.pad(next_emissions[1:], (0, 1), value=neg_inf)
        candidates.append(advance)
        skip = F.pad(following[2:], (0, 2), value=neg_inf)
        skip = skip + F.pad(next_emissions[2:], (0, 2), value=neg_inf)
        skip = torch.where(skip_mask, skip, neg_inf)
        candidates.append(skip)
        beta[step] = torch.logsumexp(torch.stack(candidates, dim=0), dim=0)

    terminal_states = [alpha[-1, states - 1]]
    if states > 1 and extended[-2] != blank_id:
        terminal_states.append(alpha[-1, states - 2])
    log_z = torch.logsumexp(torch.stack(terminal_states), dim=0)
    gamma_state = (alpha + beta - log_z).exp()
    gamma_class = log_probs.new_zeros((time_steps, num_classes))
    gamma_class.index_add_(1, extended, gamma_state)
    return gamma_class


def ctc_viterbi_path(log_probs, target, blank_id=0):
    """Return the class id visited by the best legal CTC path at each frame."""
    time_steps, num_classes = log_probs.shape
    if target.numel() == 0:
        if time_steps == 0:
            return target.new_empty((0,))
        return target.new_full((time_steps,), blank_id)
    extended = _extend_targets(target.to(log_probs.device), blank_id=blank_id)
    states = extended.numel()
    if target.numel() > time_steps:
        raise ValueError("CTC target cannot fit in the available frame count")
    repeated = (target[1:] == target[:-1]).sum().item()
    if target.numel() + repeated > time_steps:
        raise ValueError("CTC target cannot fit in the available frame count")

    neg_inf = log_probs.new_tensor(-1e30)
    skip_mask = torch.tensor(
        [
            bool(_can_skip(extended, state, blank_id))
            for state in range(states)
        ],
        dtype=torch.bool,
        device=log_probs.device,
    )
    score = log_probs.new_full((time_steps, states), -1e30)
    backpointer = torch.full(
        (time_steps, states),
        -1,
        dtype=torch.long,
        device=log_probs.device,
    )
    score[0, 0] = log_probs[0, blank_id]
    if states > 1:
        score[0, 1] = log_probs[0, extended[1]]

    for step in range(1, time_steps):
        previous = score[step - 1]
        stay_score = previous
        advance_score = F.pad(previous[:-1], (1, 0), value=neg_inf)
        skip_score = F.pad(previous[:-2], (2, 0), value=neg_inf)
        destination_skip_mask = F.pad(
            skip_mask[:-2], (2, 0), value=False
        )
        skip_score = torch.where(destination_skip_mask, skip_score, neg_inf)
        candidates = torch.stack(
            [stay_score, advance_score, skip_score], dim=0
        )
        best_score, best_move = candidates.max(dim=0)
        score[step] = best_score + log_probs[step, extended]
        state_ids = torch.arange(states, device=log_probs.device)
        backpointer[step] = torch.where(
            best_move == 0,
            state_ids,
            torch.where(
                best_move == 1,
                state_ids - 1,
                state_ids - 2,
            ),
        )

    terminal_states = [states - 1]
    if states > 1 and extended[-2] != blank_id:
        terminal_states.append(states - 2)
    terminal_scores = score[-1, terminal_states]
    final_state = terminal_states[int(terminal_scores.argmax().item())]
    path_states = torch.empty(
        time_steps, dtype=torch.long, device=log_probs.device
    )
    state = final_state
    for step in range(time_steps - 1, -1, -1):
        path_states[step] = state
        if step > 0:
            state = int(backpointer[step, state].item())
    return extended[path_states]


class CTCPrototypeHead(nn.Module):
    """CTC-aligned class prototypes with EMA statistics and a C+1 cosine head."""

    def __init__(
        self,
        num_classes,
        feature_dim,
        momentum=0.9,
        confidence=0.5,
        temperature=0.1,
        min_weight=0.05,
        distribution="cosine",
        variance_shrinkage=0.1,
        variance_floor=1e-4,
    ):
        super().__init__()
        if distribution not in {"cosine", "gaussian"}:
            raise ValueError("distribution must be cosine or gaussian")
        self.num_classes = num_classes
        self.momentum = momentum
        self.confidence = confidence
        self.temperature = temperature
        self.min_weight = min_weight
        self.distribution = distribution
        self.variance_shrinkage = float(variance_shrinkage)
        self.variance_floor = float(variance_floor)
        if not 0.0 <= self.variance_shrinkage <= 1.0:
            raise ValueError("variance_shrinkage must be between zero and one")
        if self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("support", torch.zeros(num_classes))
        self.register_buffer(
            "second_moment", torch.zeros(num_classes, feature_dim)
        )
        self.register_buffer(
            "variances", torch.ones(num_classes, feature_dim)
        )
        self.register_buffer("global_variance", torch.ones(feature_dim))
        self.register_buffer("global_support", torch.zeros(()))

    @torch.no_grad()
    def update(self, features, gamma, confidence):
        """Update prototypes from [T,B,D], [T,B,C], and [T,B] tensors."""
        confidence_gate = (confidence >= self.confidence).to(gamma.dtype)
        weighted = gamma * confidence.unsqueeze(-1) * confidence_gate.unsqueeze(-1)
        total = weighted.sum(dim=(0, 1))
        numerator = torch.einsum("tbc,tbd->cd", weighted, features)
        observed = numerator / total.clamp_min(1e-8).unsqueeze(-1)
        valid = total >= self.min_weight
        seen = self.support > 0
        blend = torch.where(seen, self.momentum, observed.new_tensor(0.0))
        updated = (
            self.prototypes * blend.unsqueeze(-1)
            + observed * (1.0 - blend).unsqueeze(-1)
        )
        self.prototypes.copy_(
            torch.where(valid.unsqueeze(-1), updated, self.prototypes)
        )
        self.support.add_(torch.where(valid, total, total.new_zeros(())))
        if self.distribution == "gaussian":
            observed_second = torch.einsum(
                "tbc,tbd->cd", weighted, features.square()
            ) / total.clamp_min(1e-8).unsqueeze(-1)
            second_blend = torch.where(
                seen, self.momentum, observed_second.new_tensor(0.0)
            )
            updated_second = (
                self.second_moment * second_blend.unsqueeze(-1)
                + observed_second * (1.0 - second_blend).unsqueeze(-1)
            )
            self.second_moment.copy_(torch.where(
                valid.unsqueeze(-1), updated_second, self.second_moment
            ))
            residual_sum = (
                observed_second
                - 2.0 * observed * observed
                + observed.square()
            )
            total_support = total.sum()
            if total_support >= self.min_weight:
                global_observed = (
                    residual_sum.sum(dim=0) / total_support
                ).clamp_min(self.variance_floor)
                global_blend = (
                    self.momentum
                    if self.global_support >= self.min_weight
                    else 0.0
                )
                self.global_variance.mul_(global_blend).add_(
                    global_observed, alpha=1.0 - global_blend
                )
                self.global_support.add_(total_support)
            class_variance = (
                self.second_moment - self.prototypes.square()
            ).clamp_min(self.variance_floor)
            class_variance = (
                (1.0 - self.variance_shrinkage) * class_variance
                + self.variance_shrinkage
                * self.global_variance.view(1, -1)
            )
            self.variances.copy_(torch.where(
                valid.unsqueeze(-1), class_variance, self.variances
            ))

    def forward(self, features):
        valid = (self.support >= self.min_weight).view(1, 1, -1)
        if self.distribution == "cosine":
            feature_norm = F.normalize(features, dim=-1, eps=1e-6)
            prototype_norm = F.normalize(self.prototypes, dim=-1, eps=1e-6)
            logits = (
                torch.matmul(feature_norm, prototype_norm.t())
                / self.temperature
            )
            return logits.masked_fill(~valid, 0.0)

        inverse_variance = self.variances.clamp_min(
            self.variance_floor
        ).reciprocal()
        feature_quadratic = torch.matmul(
            features.square(), inverse_variance.t()
        )
        cross_term = torch.matmul(
            features, (self.prototypes * inverse_variance).t()
        )
        prototype_quadratic = (
            self.prototypes.square() * inverse_variance
        ).sum(dim=-1).view(1, 1, -1)
        log_determinant = inverse_variance.log().neg().sum(
            dim=-1
        ).view(1, 1, -1)
        squared_distance = (
            feature_quadratic
            - 2.0 * cross_term
            + prototype_quadratic
            + log_determinant
        )
        logits = -0.5 * squared_distance / features.size(-1)
        return logits.masked_fill(~valid, 0.0)

    def alignment_loss(self, prototype_logits, gamma, confidence):
        valid = (self.support >= self.min_weight).view(1, 1, -1)
        valid_gamma = gamma * valid.to(gamma.dtype)
        target_mass = valid_gamma.sum(dim=-1)
        log_probs = prototype_logits.masked_fill(
            ~valid, -1e4
        ).log_softmax(dim=-1)
        normalized_gamma = valid_gamma / target_mass.clamp_min(1e-8).unsqueeze(-1)
        per_frame = -(normalized_gamma * log_probs).sum(dim=-1)
        mask = (confidence >= self.confidence) & (target_mass > 0.0)
        if not mask.any():
            return prototype_logits.sum() * 0.0
        return per_frame[mask].mean()


def collect_batch_posteriors(
    logits, labels, label_lengths, feature_lengths=None, blank_id=0
):
    """Collect base-CTC posterior occupancy for a padded [T,B,C] batch."""
    gammas = []
    confidences = []
    offset = 0
    for batch_idx, label_length in enumerate(label_lengths.tolist()):
        if feature_lengths is None:
            sequence_length = logits.size(0)
        else:
            sequence_length = int(feature_lengths[batch_idx].item())
            sequence_length = max(0, min(sequence_length, logits.size(0)))
        target = labels[offset : offset + label_length]
        offset += label_length
        gamma = logits.new_zeros((logits.size(0), logits.size(2)))
        confidence = logits.new_zeros(logits.size(0))
        if sequence_length:
            log_probs = logits[:sequence_length, batch_idx].detach().log_softmax(dim=-1)
            try:
                valid_gamma = ctc_posterior(
                    log_probs, target, blank_id=blank_id
                )
            except ValueError:
                # The CTC loss path excludes impossible samples before
                # backward, so their prototype occupancy must be omitted too.
                gammas.append(gamma)
                confidences.append(confidence)
                continue
            gamma[:sequence_length] = valid_gamma
            top_two = valid_gamma.topk(2, dim=-1).values
            confidence[:sequence_length] = top_two[:, 0] - top_two[:, 1]
        gammas.append(gamma)
        confidences.append(confidence)
    return torch.stack(gammas, dim=1), torch.stack(confidences, dim=1)
