import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RDB(nn.Module):
    """Latent temporal prototype reconstruction with signer control.

    This module augments latent CTC features. It does not generate pixels or
    claim to synthesize real videos. The reconstruction decoder receives a
    supported gloss-prototype mixture, a short-time motion residual, and an
    optional signer style vector. Its output projection is zero initialized,
    so the initial augmentation is an exact identity.
    """

    def __init__(
        self,
        feature_dim,
        max_strength=0.20,
        hidden_size=64,
        kernel_size=3,
        min_content_mass=0.05,
        min_prototype_support=0.05,
        confidence_threshold=0.1,
        blank_id=0,
        style_separation_margin=1e-4,
        reconstruction_mse_weight=1.0,
        reconstruction_cosine_weight=0.1,
        reconstruction_temperature=0.1,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        hidden_size = int(hidden_size)
        kernel_size = int(kernel_size)
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        max_strength = float(max_strength)
        if not math.isfinite(max_strength) or max_strength <= 0.0:
            raise ValueError("max_strength must be finite and positive")
        self.feature_dim = feature_dim
        self.max_strength = max_strength
        self.kernel_size = kernel_size
        self.min_content_mass = self._probability(
            min_content_mass, "min_content_mass"
        )
        self.min_prototype_support = self._probability(
            min_prototype_support, "min_prototype_support"
        )
        self.confidence_threshold = self._probability(
            confidence_threshold, "confidence_threshold"
        )
        self.blank_id = int(blank_id)
        self.style_separation_margin = float(style_separation_margin)
        self.reconstruction_mse_weight = float(reconstruction_mse_weight)
        self.reconstruction_cosine_weight = float(reconstruction_cosine_weight)
        self.reconstruction_temperature = float(reconstruction_temperature)
        if (
            not math.isfinite(self.style_separation_margin)
            or self.style_separation_margin < 0.0
        ):
            raise ValueError(
                "style_separation_margin must be finite and non-negative"
            )
        for name, value in (
            ("reconstruction_mse_weight", self.reconstruction_mse_weight),
            (
                "reconstruction_cosine_weight",
                self.reconstruction_cosine_weight,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative".format(name))
        if (
            not math.isfinite(self.reconstruction_temperature)
            or self.reconstruction_temperature <= 0.0
        ):
            raise ValueError(
                "reconstruction_temperature must be finite and positive"
            )
        if self.reconstruction_mse_weight == 0.0 and (
            self.reconstruction_cosine_weight == 0.0
        ):
            raise ValueError("at least one reconstruction term must be non-zero")

        input_dim = 3 * feature_dim + 2 * feature_dim + 2
        self.pointwise = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        self.temporal = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
        )
        self.output = nn.Conv1d(hidden_size, feature_dim, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        # A non-zero bounded gate lets gradients reach the zero-initialized
        # output projection while the projected update itself remains zero.
        self.strength_raw = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _probability(value, name):
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("{} must be between zero and one".format(name))
        return value

    def strength(self):
        return self.max_strength * torch.tanh(self.strength_raw)

    def _validity_and_lengths(self, features, lengths):
        time_steps, batch_size = features.shape[:2]
        if lengths is None:
            lengths = features.new_full((batch_size,), time_steps, dtype=torch.long)
        else:
            if lengths.ndim != 1 or lengths.numel() != batch_size:
                raise ValueError("lengths must have shape [B]")
            if not torch.isfinite(lengths).all():
                raise ValueError("lengths must contain finite values")
            lengths = lengths.to(device=features.device, dtype=torch.long)
        if (lengths < 0).any() or (lengths > time_steps).any():
            raise ValueError("lengths must be between zero and T")
        time_ids = torch.arange(time_steps, device=features.device).view(-1, 1)
        valid = time_ids < lengths.view(1, -1)
        return lengths, valid

    def _prototype_content(
        self,
        features,
        prototypes,
        support,
        gamma,
        confidence,
        valid,
    ):
        batch_size = features.size(1)
        if prototypes.ndim != 2 or prototypes.size(1) != features.size(2):
            raise ValueError("prototypes must have shape [C,D]")
        num_prototypes = prototypes.size(0)
        if support.reshape(-1).numel() != num_prototypes:
            raise ValueError("support must contain one value per prototype")
        if gamma.shape != (
            features.size(0),
            batch_size,
            num_prototypes,
        ):
            raise ValueError(
                "gamma must have shape [T,B,C] matching features and prototypes"
            )
        if confidence.shape != features.shape[:2]:
            raise ValueError("confidence must have shape [T,B]")
        if not 0 <= self.blank_id < num_prototypes:
            raise ValueError("blank_id is outside the prototype range")

        class_ids = torch.arange(
            num_prototypes, device=features.device
        ) != self.blank_id
        supported = (
            support.reshape(-1).to(features.device)
            >= self.min_prototype_support
        )
        allowed = class_ids & supported
        class_gamma = gamma[..., class_ids].to(features.dtype)
        class_gamma = class_gamma * allowed[class_ids].to(class_gamma.dtype)
        mass = class_gamma.sum(dim=-1)
        active = (
            valid
            & (mass >= self.min_content_mass)
            & (
                confidence.to(features.device)
                >= self.confidence_threshold
            )
        )
        normalized = class_gamma / mass.clamp_min(1e-8).unsqueeze(-1)
        content_mix = torch.einsum(
            "tbc,cd->tbd",
            normalized,
            prototypes[class_ids].to(features.dtype),
        )
        return content_mix, mass, active

    def _temporal_residual(self, features, valid):
        channels_first = features.permute(1, 2, 0)
        frame_mask = valid.transpose(0, 1).unsqueeze(1).to(features.dtype)
        masked_features = channels_first * frame_mask
        padding = self.kernel_size // 2
        smoothed_sum = F.avg_pool1d(
            F.pad(masked_features, (padding, padding)),
            kernel_size=self.kernel_size,
            stride=1,
        )
        denominator = F.avg_pool1d(
            F.pad(frame_mask, (padding, padding)),
            kernel_size=self.kernel_size,
            stride=1,
        ).clamp_min(1e-6)
        smoothed = (smoothed_sum / denominator).permute(2, 0, 1)
        return torch.where(valid.unsqueeze(-1), features - smoothed, 0.0)

    def _decode(
        self,
        features,
        content_mix,
        temporal_residual,
        style,
        mass,
        confidence,
        valid,
    ):
        time_steps, batch_size, feature_dim = features.shape
        if style is None:
            style = features.new_zeros((batch_size, 2 * feature_dim))
        style_frames = style.to(
            device=features.device, dtype=features.dtype
        ).unsqueeze(0).expand(time_steps, -1, -1)
        quality = torch.stack(
            (
                mass.to(features.dtype),
                confidence.to(features.device).to(features.dtype),
            ),
            dim=-1,
        )
        decoder_input = torch.cat(
            (
                features,
                content_mix,
                temporal_residual,
                style_frames,
                quality,
            ),
            dim=-1,
        )
        decoder_input = decoder_input * valid.unsqueeze(-1).to(
            decoder_input.dtype
        )
        decoder_input = decoder_input.transpose(1, 2)
        hidden = F.gelu(self.pointwise(decoder_input))
        hidden = F.gelu(self.temporal(hidden))
        bounded_delta = torch.tanh(self.output(hidden).transpose(1, 2))
        return bounded_delta

    def _style_separation_loss(self, features, counterfactual, active, mask):
        if counterfactual is None or mask is None or not mask.any():
            return features.sum() * 0.0
        frame_mask = active & mask.view(1, -1)
        if not frame_mask.any():
            return features.sum() * 0.0
        difference = counterfactual - features
        difference_rms = (
            difference.square().mean(dim=-1) + 1e-8
        ).sqrt()
        reference_rms = (
            features.detach().square().mean(dim=-1) + 1e-8
        ).sqrt()
        relative_shift = difference_rms / (reference_rms + 1e-6)
        shortfall = F.relu(self.style_separation_margin - relative_shift)
        return shortfall[frame_mask].mean()

    def forward(
        self,
        features,
        prototypes,
        support,
        gamma,
        confidence,
        lengths=None,
        style=None,
        counterfactual_style=None,
        style_swap_mask=None,
    ):
        if features.ndim != 3:
            raise ValueError("features must have shape [T,B,D]")
        if not torch.isfinite(features).all():
            raise ValueError("features must contain finite values")
        if features.size(2) != self.feature_dim:
            raise ValueError("features have the wrong feature dimension")
        for name, tensor in (
            ("prototypes", prototypes),
            ("support", support),
            ("gamma", gamma),
            ("confidence", confidence),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError("{} must contain finite values".format(name))

        _, valid = self._validity_and_lengths(features, lengths)
        content_mix, mass, active = self._prototype_content(
            features,
            prototypes,
            support,
            gamma,
            confidence,
            valid,
        )
        temporal_residual = self._temporal_residual(features, valid)
        if style is not None:
            expected_style_shape = (features.size(1), 2 * features.size(2))
            if style.shape != expected_style_shape:
                raise ValueError("style must have shape [B,2D]")
            if not torch.isfinite(style).all():
                raise ValueError("style must contain finite values")
        if counterfactual_style is not None:
            expected_style_shape = (features.size(1), 2 * features.size(2))
            if counterfactual_style.shape != expected_style_shape:
                raise ValueError("counterfactual_style must have shape [B,2D]")
            if not torch.isfinite(counterfactual_style).all():
                raise ValueError(
                    "counterfactual_style must contain finite values"
                )
            if style is None:
                raise ValueError(
                    "counterfactual_style requires the original style"
                )

        strength = self.strength()
        active_frames = active.unsqueeze(-1)
        residual = self._decode(
            features,
            content_mix,
            temporal_residual,
            style,
            mass,
            confidence,
            valid,
        )
        delta = (
            strength
            * active_frames.to(residual.dtype)
            * residual
        )
        reconstructed = features + delta

        counterfactual = None
        style_shift = features.sum() * 0.0
        separation_loss = features.sum() * 0.0
        if counterfactual_style is not None:
            if style_swap_mask is None:
                style_swap_mask = torch.ones(
                    features.size(1),
                    dtype=torch.bool,
                    device=features.device,
                )
            else:
                if style_swap_mask.ndim != 1 or style_swap_mask.numel() != features.size(1):
                    raise ValueError("style_swap_mask must have shape [B]")
                style_swap_mask = style_swap_mask.to(
                    device=features.device, dtype=torch.bool
                )
            counterfactual_residual = self._decode(
                features,
                content_mix,
                temporal_residual,
                counterfactual_style,
                mass,
                confidence,
                valid,
            )
            counterfactual_delta = (
                strength
                * active_frames.to(counterfactual_residual.dtype)
                * counterfactual_residual
            )
            counterfactual_delta = counterfactual_delta * style_swap_mask.view(
                1, -1, 1
            ).to(counterfactual_delta.dtype)
            counterfactual = features + counterfactual_delta
            style_frames = active & style_swap_mask.view(1, -1)
            if style_frames.any():
                style_shift = (
                    counterfactual - features
                ).abs()[style_frames].mean()
            separation_loss = self._style_separation_loss(
                features,
                counterfactual,
                active,
                style_swap_mask,
            )

        active_count = active.to(features.dtype).sum().clamp_min(1.0)
        mean_shift = (
            delta.abs() * active_frames.to(delta.dtype)
        ).sum() / (active_count * features.size(2))
        return {
            "features": reconstructed,
            "counterfactual_features": counterfactual,
            "content_mix": content_mix,
            "temporal_residual": temporal_residual,
            "active": active,
            "content_mass": mass,
            "strength": strength,
            "mean_shift": mean_shift,
            "style_shift": style_shift,
            "style_separation_loss": separation_loss,
        }

    def reconstruction_loss(
        self,
        target,
        reconstructed,
        lengths=None,
        active=None,
    ):
        if target.shape != reconstructed.shape or target.ndim != 3:
            raise ValueError("target and reconstructed must have shape [T,B,D]")
        if not torch.isfinite(target).all() or not torch.isfinite(
            reconstructed
        ).all():
            raise ValueError("reconstruction inputs must be finite")
        _, valid = self._validity_and_lengths(target, lengths)
        mask = valid
        if active is not None:
            if active.shape != valid.shape:
                raise ValueError("active must have shape [T,B]")
            mask = mask & active.to(device=valid.device, dtype=torch.bool)
        if not mask.any():
            zero = reconstructed.sum() * 0.0
            return {"loss": zero, "mse": zero, "cosine": zero}

        target = target.detach()
        frame_mask = mask.unsqueeze(-1)
        squared_error = (reconstructed - target).square()
        mse = squared_error[frame_mask.expand_as(squared_error)].mean()
        cosine = (
            (
                1.0
                - F.cosine_similarity(
                    reconstructed[mask],
                    target[mask],
                    dim=-1,
                    eps=1e-6,
                )
            ).clamp_min(0.0)
            / self.reconstruction_temperature
        )
        cosine = cosine.mean()
        loss = (
            self.reconstruction_mse_weight * mse
            + self.reconstruction_cosine_weight * cosine
        )
        return {"loss": loss, "mse": mse, "cosine": cosine}
