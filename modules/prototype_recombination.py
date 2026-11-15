import math

import torch
import torch.nn as nn


class PrototypeRecombination(nn.Module):
    """Class-preserving CTC-prototype feature augmentation.

    Every frame moves toward a posterior-weighted mixture of supported gloss
    prototypes. The raw feature and target prototype share the same dimension,
    and the original gloss sequence remains the CTC target of the augmented
    branch. The learned strength is bounded and starts at exact identity.
    """

    def __init__(
        self,
        max_strength=0.25,
        min_mix_mass=0.05,
        min_prototype_support=0.05,
        confidence_threshold=0.0,
        blank_id=0,
    ):
        super().__init__()
        max_strength = float(max_strength)
        if not math.isfinite(max_strength) or max_strength <= 0.0:
            raise ValueError("max_strength must be finite and positive")
        self.max_strength = max_strength
        self.min_mix_mass = float(min_mix_mass)
        self.min_prototype_support = float(min_prototype_support)
        self.confidence_threshold = float(confidence_threshold)
        self.blank_id = int(blank_id)
        if not 0.0 <= self.min_mix_mass <= 1.0:
            raise ValueError("min_mix_mass must be between zero and one")
        if not 0.0 <= self.min_prototype_support <= 1.0:
            raise ValueError(
                "min_prototype_support must be between zero and one"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between zero and one"
            )
        self.strength_raw = nn.Parameter(torch.zeros(()))

    def strength(self):
        return self.max_strength * torch.tanh(self.strength_raw)

    def forward(
        self,
        features,
        prototypes,
        support,
        gamma=None,
        confidence=None,
    ):
        if features.ndim != 3:
            raise ValueError("features must have shape [T,B,D]")
        zero = features.sum() * 0.0
        strength = self.strength()
        if gamma is None:
            return {
                "features": features,
                "prototype_mix": None,
                "active": torch.zeros(
                    features.shape[:2],
                    dtype=torch.bool,
                    device=features.device,
                ),
                "strength": strength,
                "mean_shift": zero,
            }
        if gamma.shape[:2] != features.shape[:2]:
            raise ValueError("gamma must match the feature time and batch axes")
        if prototypes.shape != (
            gamma.size(-1),
            features.size(-1),
        ):
            raise ValueError("prototypes must be [C,D] and match gamma classes")
        if support.reshape(-1).numel() != prototypes.size(0):
            raise ValueError("support must contain one value per prototype")
        if not 0 <= self.blank_id < prototypes.size(0):
            raise ValueError("blank_id is outside the prototype range")

        class_ids = torch.arange(
            prototypes.size(0), device=features.device
        ) != self.blank_id
        valid = class_ids & (
            support.reshape(-1).to(features.device)
            >= self.min_prototype_support
        )
        class_gamma = gamma[..., class_ids].to(features.dtype)
        class_gamma = class_gamma * valid[class_ids].to(class_gamma.dtype)
        mass = class_gamma.sum(dim=-1)
        active = mass >= self.min_mix_mass
        if confidence is not None:
            if confidence.shape != features.shape[:2]:
                raise ValueError("confidence must have shape [T,B]")
            active = active & (
                confidence.to(features.device)
                >= self.confidence_threshold
            )
        normalized = class_gamma / mass.clamp_min(1e-8).unsqueeze(-1)
        prototype_mix = torch.einsum(
            "tbc,cd->tbd",
            normalized,
            prototypes[class_ids].to(features.dtype),
        )
        active = active.unsqueeze(-1)
        delta = prototype_mix - features
        augmented = features + strength * active.to(features.dtype) * delta
        shift = (augmented - features).abs().mean()
        return {
            "features": augmented,
            "prototype_mix": prototype_mix,
            "active": active.squeeze(-1),
            "strength": strength,
            "mean_shift": shift,
        }
