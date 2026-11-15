import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy_from_log_probs(log_probs):
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def _top_margin_from_log_probs(log_probs):
    probabilities = log_probs.exp()
    top_two = probabilities.topk(2, dim=-1).values
    return top_two[..., 0] - top_two[..., 1]


def _entropy_from_logits(logits):
    return _entropy_from_log_probs(logits.log_softmax(dim=-1))


def _top_margin(logits):
    return _top_margin_from_log_probs(logits.log_softmax(dim=-1))


class FDGCCalibrator(nn.Module):
    """Frequency-difficulty gated calibration on a full C+1 CTC emission."""

    def __init__(
        self,
        num_classes,
        hidden_size=8,
        tau_max=2.0,
        delta_max=0.5,
        blank_bias_max=1.0,
        blank_id=0,
        use_frequency=True,
        use_difficulty=True,
    ):
        super().__init__()
        if not 0 <= blank_id < num_classes:
            raise ValueError("blank_id must index the classifier output")
        self.num_classes = num_classes
        self.blank_id = blank_id
        self.tau_max = float(tau_max)
        self.delta_max = float(delta_max)
        self.blank_bias_max = float(blank_bias_max)
        self.use_frequency = bool(use_frequency)
        self.use_difficulty = bool(use_difficulty)

        self.class_mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.blank_logit = nn.Parameter(torch.zeros(()))
        self._zero_initialize_outputs()

    def _zero_initialize_outputs(self):
        nn.init.zeros_(self.class_mlp[-1].weight)
        nn.init.zeros_(self.class_mlp[-1].bias)
        nn.init.zeros_(self.frame_mlp[-1].weight)
        nn.init.zeros_(self.frame_mlp[-1].bias)

    def forward(
        self,
        logits,
        source_prior,
        target_prior=None,
        difficulty=None,
        frame_mask=None,
        input_is_log_probs=False,
    ):
        """Calibrate [T,B,C+1] logits while preserving per-frame normalization."""
        if logits.shape[-1] != self.num_classes:
            raise ValueError("logits must have num_classes channels")
        source_prior = source_prior.to(
            device=logits.device, dtype=logits.dtype
        )
        expected = self.num_classes - 1
        if source_prior.numel() != expected:
            raise ValueError(
                f"source_prior must contain {expected} non-blank classes"
            )
        source_prior = source_prior.clamp_min(torch.finfo(logits.dtype).tiny)
        source_prior = source_prior / source_prior.sum()
        if target_prior is None:
            target_prior = torch.full_like(source_prior, 1.0 / expected)
        else:
            target_prior = target_prior.to(
                device=logits.device, dtype=logits.dtype
            )
            if target_prior.numel() != expected:
                raise ValueError(
                    f"target_prior must contain {expected} non-blank classes"
                )
            target_prior = target_prior.clamp_min(
                torch.finfo(logits.dtype).tiny
            )
            target_prior = target_prior / target_prior.sum()
        if difficulty is None:
            difficulty = torch.zeros_like(source_prior)
        else:
            difficulty = difficulty.to(
                device=logits.device, dtype=logits.dtype
            )
            if difficulty.numel() != expected:
                raise ValueError(
                    f"difficulty must contain {expected} non-blank classes"
                )

        frequency_input = (
            source_prior.log()
            if self.use_frequency
            else torch.zeros_like(source_prior)
        )
        difficulty_input = (
            difficulty.clamp(0.0, 1.0)
            if self.use_difficulty
            else torch.zeros_like(difficulty)
        )
        class_input = torch.stack(
            [frequency_input, difficulty_input],
            dim=-1,
        )
        class_logit = self.class_mlp(class_input).squeeze(-1)
        tau_class = self.tau_max * (
            F.softplus(class_logit) - math.log(2.0)
        ).clamp(0.0, 1.0)

        input_log_probs = (
            logits if input_is_log_probs else logits.log_softmax(dim=-1)
        )
        entropy = _entropy_from_log_probs(input_log_probs)
        margin = _top_margin_from_log_probs(input_log_probs)
        frame_input = torch.stack([entropy, margin], dim=-1)
        frame_delta = self.delta_max * torch.tanh(
            self.frame_mlp(frame_input).squeeze(-1)
        )
        tau = (
            tau_class.view(1, 1, -1) + frame_delta.unsqueeze(-1)
        ).clamp(0.0, self.tau_max)

        blank_bias = self.blank_bias_max * torch.tanh(self.blank_logit)
        if self.use_frequency:
            log_ratio = target_prior.log() - source_prior.log()
            class_adjustment = tau * log_ratio.view(1, 1, -1)
        else:
            class_adjustment = torch.zeros_like(tau)

        nonblank_mask = torch.ones(
            self.num_classes, dtype=torch.bool, device=logits.device
        )
        nonblank_mask[self.blank_id] = False
        adjusted = logits.clone()
        adjusted[..., nonblank_mask] = (
            adjusted[..., nonblank_mask] + class_adjustment
        )
        adjusted[..., self.blank_id] = (
            adjusted[..., self.blank_id] + blank_bias
        )

        if frame_mask is not None:
            frame_mask = frame_mask.to(device=logits.device, dtype=torch.bool)
            if frame_mask.shape != logits.shape[:2]:
                raise ValueError("frame_mask must have shape [T,B]")
            adjusted = torch.where(
                frame_mask.unsqueeze(-1), adjusted, logits
            )
            tau = torch.where(frame_mask.unsqueeze(-1), tau, torch.zeros_like(tau))

        return {
            "logits": adjusted,
            "log_probs": adjusted.log_softmax(dim=-1),
            "tau": tau,
            "tau_class": tau_class,
            "blank_bias": blank_bias,
        }

    def calibration_loss(
        self,
        output,
        target_classes,
        frame_mask,
        regularization=1e-3,
    ):
        """Frame NLL plus bounded-strength regularization on calibrated frames."""
        frame_mask = frame_mask.to(
            device=target_classes.device, dtype=torch.bool
        )
        if not frame_mask.any():
            return output["logits"].sum() * 0.0
        log_probs = output["log_probs"][frame_mask]
        targets = target_classes[frame_mask]
        nll = F.nll_loss(log_probs, targets)
        tau = output["tau"][frame_mask]
        return nll + float(regularization) * tau.square().mean()


FUSION_PARAMETERIZATION_VERSION = 2


def validate_fusion_state(state_dict):
    """Reject calibration state produced with a different fusion geometry."""
    if "fusion_parameterization_version" not in state_dict:
        raise ValueError(
            "multichannel checkpoint has no fusion parameterization version"
        )
    version = state_dict["fusion_parameterization_version"]
    if torch.is_tensor(version):
        version = version.detach().cpu().item()
    if int(version) != FUSION_PARAMETERIZATION_VERSION:
        raise ValueError(
            "multichannel fusion parameterization version {} is incompatible "
            "with version {}".format(
                version,
                FUSION_PARAMETERIZATION_VERSION,
            )
        )


class MultiChannelCalibrator(nn.Module):
    """Temperature-scale channels around a base-linear probability anchor."""

    def __init__(
        self,
        num_classes,
        min_temperature=0.25,
        max_temperature=4.0,
        linear_anchor_gate_init=-6.0,
        fusion_regularization=1e-2,
    ):
        super().__init__()
        if min_temperature <= 0.0:
            raise ValueError("min_temperature must be positive")
        if max_temperature < min_temperature:
            raise ValueError("max_temperature must not be smaller than min")
        fusion_regularization = float(fusion_regularization)
        if fusion_regularization < 0.0:
            raise ValueError("fusion_regularization must be non-negative")
        self.num_classes = int(num_classes)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.fusion_regularization = fusion_regularization
        self.log_temperatures = nn.Parameter(torch.zeros(3))
        # fusion_logits[0] controls how much probability mass may leave the
        # linear channel. The remaining two logits split that residual mass.
        self.fusion_logits = nn.Parameter(
            torch.tensor([float(linear_anchor_gate_init), 0.0, 0.0])
        )
        self.register_buffer(
            "fusion_parameterization_version",
            torch.tensor(
                FUSION_PARAMETERIZATION_VERSION,
                dtype=torch.long,
            ),
        )

    def temperatures(self):
        return self.log_temperatures.exp().clamp(
            self.min_temperature,
            self.max_temperature,
        )

    def weights(self):
        nonlinear_mass = torch.sigmoid(self.fusion_logits[0])
        nonlinear_weights = (
            self.fusion_logits[1:].softmax(dim=0) * nonlinear_mass
        )
        return torch.cat(
            [1.0 - nonlinear_mass.view(1), nonlinear_weights],
            dim=0,
        )

    def linear_anchor_penalty(self):
        return torch.sigmoid(self.fusion_logits[0]).square()

    def forward(self, linear_logits, visual_logits, text_logits):
        channels = (linear_logits, visual_logits, text_logits)
        expected_shape = linear_logits.shape
        if expected_shape[-1] != self.num_classes:
            raise ValueError("channel logits have an unexpected class count")
        for channel in channels:
            if channel.shape != expected_shape:
                raise ValueError("all channel logits must have the same shape")
            if not torch.isfinite(channel).all():
                raise FloatingPointError("channel logits must be finite")

        temperatures = self.temperatures().to(
            device=linear_logits.device,
            dtype=linear_logits.dtype,
        )
        weights = self.weights().to(
            device=linear_logits.device,
            dtype=linear_logits.dtype,
        )
        scaled_probabilities = []
        for channel_index, channel in enumerate(channels):
            scaled = channel / temperatures[channel_index]
            scaled_probabilities.append(scaled.softmax(dim=-1))
        fused = (
            weights[0] * scaled_probabilities[0]
            + weights[1] * scaled_probabilities[1]
            + weights[2] * scaled_probabilities[2]
        )
        fused = fused.clamp_min(torch.finfo(fused.dtype).tiny)
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(fused.dtype).tiny
        )
        return {
            "probabilities": fused,
            "log_probs": fused.log(),
            "temperatures": temperatures,
            "weights": weights,
        }

    def calibration_loss(
        self,
        output,
        target_classes,
        frame_mask,
        regularization=1e-4,
    ):
        frame_mask = frame_mask.to(
            device=output["log_probs"].device,
            dtype=torch.bool,
        )
        if frame_mask.shape != output["log_probs"].shape[:2]:
            raise ValueError("frame_mask must have shape [T,B]")
        if not frame_mask.any():
            return output["log_probs"].sum() * 0.0
        log_probs = output["log_probs"][frame_mask]
        targets = target_classes.to(log_probs.device)[frame_mask]
        nll = F.nll_loss(log_probs, targets)
        temperature_penalty = (
            self.temperatures() - 1.0
        ).square().mean()
        return (
            nll
            + float(regularization) * temperature_penalty
            + self.fusion_regularization * self.linear_anchor_penalty()
        )
