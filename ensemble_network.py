import importlib
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import utils


def import_class(name):
    module_name, class_name = str(name).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def class_prior_adjustment(probabilities, prior, tau):
    tau = float(tau)
    if tau < 0.0:
        raise ValueError("prior tau must be non-negative")
    if tau == 0.0:
        return probabilities
    prior = torch.as_tensor(
        prior,
        dtype=probabilities.dtype,
        device=probabilities.device,
    ).flatten()
    if prior.numel() != probabilities.size(-1):
        raise ValueError("class prior must match the classifier output")
    if not torch.isfinite(prior).all() or (prior < 0.0).any():
        raise ValueError("class prior must contain finite non-negative values")
    factors = torch.ones_like(prior)
    nonblank = prior[1:].clamp_min(torch.finfo(prior.dtype).tiny)
    factors[1:] = nonblank.pow(-tau)
    adjusted = probabilities * factors.view(1, -1)
    return adjusted / adjusted.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(adjusted.dtype).tiny)


def weighted_probability_average(logits, weights, temperatures):
    if not logits:
        raise ValueError("at least one member logit tensor is required")
    if len(logits) != len(weights) or len(logits) != len(temperatures):
        raise ValueError("members, weights and temperatures must align")
    combined = None
    total_weight = 0.0
    for member_logits, weight, temperature in zip(
        logits,
        weights,
        temperatures,
    ):
        weight = float(weight)
        temperature = float(temperature)
        if weight < 0.0:
            raise ValueError("member weights must be non-negative")
        if temperature <= 0.0:
            raise ValueError("member temperatures must be positive")
        if not torch.isfinite(member_logits).all():
            raise FloatingPointError("ensemble member produced non-finite logits")
        probabilities = (member_logits / temperature).softmax(dim=-1)
        weighted = weight * probabilities
        combined = weighted if combined is None else combined + weighted
        total_weight += weight
    if total_weight <= 0.0:
        raise ValueError("at least one member weight must be positive")
    combined = combined / total_weight
    return combined / combined.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def final_member_logits(member, output):
    logits = output["sequence_logits"]
    input_is_log_probs = output.get("sequence_log_probs") is not None
    if getattr(member, "multichannel_calibrator", None) is None and hasattr(
        member,
        "_prior_adjusted_logits",
    ):
        logits = member._prior_adjusted_logits(logits)
    if getattr(member, "fdgc_enabled", False):
        calibrated = member.fdgc(
            logits,
            source_prior=member._nonblank_source_prior(),
            difficulty=member.fdgc_difficulty,
            input_is_log_probs=input_is_log_probs,
        )
        logits = calibrated["logits"]
    return logits


class WeightedEnsembleModel(nn.Module):
    """Inference wrapper for calibrated heterogeneous CTC probabilities."""

    def __init__(
        self,
        num_classes,
        members,
        gloss_dict,
        loss_weights=None,
        beam_width=10,
        num_processes=10,
        class_prior_path=None,
        prior_tau=0.0,
        **unused,
    ):
        super().__init__()
        if not isinstance(members, list) or not members:
            raise ValueError("ensemble members must be a non-empty list")
        self.num_classes = int(num_classes)
        self.loss_weights = loss_weights
        self.member_names = []
        self.members = nn.ModuleDict()
        member_weights = []
        member_temperatures = []
        seen_names = set()

        for index, spec in enumerate(members):
            if not isinstance(spec, dict):
                raise ValueError("ensemble member specifications must be objects")
            name = str(spec.get("name", "")).strip()
            model_name = spec.get("model")
            if not name or not model_name:
                raise ValueError("each ensemble member needs name and model")
            if name in seen_names:
                raise ValueError("ensemble member names must be unique: " + name)
            seen_names.add(name)
            member_args = dict(spec.get("model_args", {}))
            member_args.pop("num_classes", None)
            member_args.pop("gloss_dict", None)
            member_args.pop("loss_weights", None)
            member_class = import_class(model_name)
            member = member_class(
                num_classes=self.num_classes,
                gloss_dict=gloss_dict,
                loss_weights=spec.get("loss_weights", loss_weights),
                **member_args,
            )
            self.members["member_{}".format(index)] = member
            self.member_names.append(name)
            member_weights.append(float(spec.get("weight", 1.0)))
            member_temperatures.append(float(spec.get("temperature", 1.0)))

        if sum(member_weights) <= 0.0:
            raise ValueError("at least one ensemble member needs positive weight")
        if any(weight < 0.0 for weight in member_weights):
            raise ValueError("ensemble member weights must be non-negative")
        if any(value <= 0.0 for value in member_temperatures):
            raise ValueError("ensemble member temperatures must be positive")

        self.register_buffer(
            "member_weights",
            torch.tensor(member_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "member_temperatures",
            torch.tensor(member_temperatures, dtype=torch.float32),
        )
        prior = self._load_prior(class_prior_path)
        self.register_buffer("class_prior", prior)
        self.register_buffer(
            "prior_tau",
            torch.tensor(float(prior_tau), dtype=torch.float32),
        )
        self.supports_signer = any(
            getattr(member, "supports_signer", False)
            for member in self.members.values()
        )
        self.decoder = utils.Decode(
            gloss_dict,
            self.num_classes,
            "beam",
            beam_width=int(beam_width),
            num_processes=int(num_processes),
        )

    def _load_prior(self, class_prior_path):
        if class_prior_path is None:
            return torch.zeros(self.num_classes, dtype=torch.float32)
        path = Path(class_prior_path).expanduser()
        if not path.is_file():
            raise ValueError("class prior does not exist: {}".format(path))
        prior = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
        if prior.shape != (self.num_classes,):
            raise ValueError("class prior shape does not match num_classes")
        if not np.isfinite(prior).all() or (prior < 0.0).any():
            raise ValueError("class prior must be finite and non-negative")
        return torch.from_numpy(prior.copy())

    def _run_member(self, member, video, video_length, label, label_lgt,
                    signer_ids):
        kwargs = {"label": label, "label_lgt": label_lgt}
        if "decode" in inspect.signature(member.forward).parameters:
            kwargs["decode"] = False
        if getattr(member, "supports_signer", False):
            if signer_ids is None:
                raise ValueError("ensemble member requires signer ids")
            kwargs["signer_ids"] = signer_ids
        return member(video, video_length, **kwargs)

    def _member_conv_logits(self, member, output):
        logits = output.get("conv_logits")
        if logits is None:
            raise ValueError("ensemble member did not return conv_logits")
        if hasattr(member, "_prior_adjusted_logits"):
            return member._prior_adjusted_logits(logits)
        return logits

    def forward(
        self,
        x,
        len_x,
        label=None,
        label_lgt=None,
        decode=True,
        signer_ids=None,
    ):
        if self.training:
            raise RuntimeError("WeightedEnsembleModel is inference-only")
        sequence_logits = []
        conv_logits = []
        first_output = None
        feat_len = None
        for member in self.members.values():
            output = self._run_member(
                member,
                x,
                len_x,
                label,
                label_lgt,
                signer_ids,
            )
            if feat_len is None:
                feat_len = output["feat_len"]
                first_output = output
            elif not torch.equal(feat_len, output["feat_len"]):
                raise ValueError("ensemble members produced different lengths")
            sequence_logits.append(final_member_logits(member, output))
            conv_logits.append(self._member_conv_logits(member, output))

        weights = self.member_weights.detach().cpu().tolist()
        temperatures = self.member_temperatures.detach().cpu().tolist()
        sequence_probabilities = weighted_probability_average(
            sequence_logits,
            weights,
            temperatures,
        )
        conv_probabilities = weighted_probability_average(
            conv_logits,
            weights,
            temperatures,
        )
        sequence_probabilities = class_prior_adjustment(
            sequence_probabilities,
            self.class_prior,
            float(self.prior_tau),
        )
        conv_probabilities = class_prior_adjustment(
            conv_probabilities,
            self.class_prior,
            float(self.prior_tau),
        )
        sequence_log_probs = sequence_probabilities.clamp_min(1e-8).log()
        conv_log_probs = conv_probabilities.clamp_min(1e-8).log()

        recognized_sents = None
        conv_sents = None
        if decode:
            recognized_sents = self.decoder.decode(
                sequence_log_probs,
                feat_len,
                batch_first=False,
                probs=False,
            )
            conv_sents = self.decoder.decode(
                conv_log_probs,
                feat_len,
                batch_first=False,
                probs=False,
            )

        return {
            "framewise_features": first_output["framewise_features"],
            "visual_features": first_output.get("visual_features"),
            "feat_len": feat_len,
            "conv_logits": conv_log_probs,
            "sequence_logits": sequence_log_probs,
            "conv_sents": conv_sents,
            "recognized_sents": recognized_sents,
            "ensemble_member_names": list(self.member_names),
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        raise RuntimeError("WeightedEnsembleModel cannot be trained end to end")
