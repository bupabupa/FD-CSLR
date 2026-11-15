import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.calibration import FDGCCalibrator, MultiChannelCalibrator
from modules.prototype import CTCPrototypeHead, collect_batch_posteriors
from modules.prototype_recombination import PrototypeRecombination
from modules.rdb import RDB
from modules.text_prototype import TextPrototypeHead
from modules.weighted_ctc import class_weights_from_prior
from slr_network import SLRModel


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, strength):
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def signer_multi_positive_loss(
    anchor_embeddings,
    anchor_classes,
    anchor_signers,
    memory_embeddings,
    memory_classes,
    memory_signers,
    memory_valid,
    temperature,
):
    """Cross-signer InfoNCE with a bounded detached occurrence memory."""
    zero = anchor_embeddings.sum() * 0.0
    empty_coverage = anchor_embeddings.new_tensor(0.0)
    if anchor_embeddings.numel() == 0 or not memory_valid.any():
        return zero, empty_coverage
    if temperature <= 0.0:
        raise ValueError("signer invariance temperature must be positive")

    anchors = F.normalize(anchor_embeddings, dim=-1)
    memory = F.normalize(memory_embeddings, dim=-1)
    logits = anchors @ memory.transpose(0, 1)
    logits = logits / float(temperature)

    anchor_classes = anchor_classes.to(anchor_embeddings.device).long()
    anchor_signers = anchor_signers.to(anchor_embeddings.device).long()
    memory_classes = memory_classes.to(anchor_embeddings.device).long()
    memory_signers = memory_signers.to(anchor_embeddings.device).long()
    memory_valid = memory_valid.to(anchor_embeddings.device).bool()

    valid = memory_valid.view(1, -1).expand_as(logits)
    same_class = anchor_classes.view(-1, 1) == memory_classes.view(1, -1)
    same_signer = anchor_signers.view(-1, 1) == memory_signers.view(1, -1)
    positive = valid & same_class & ~same_signer
    # Same-gloss/same-signer samples are neither positives nor hard negatives.
    candidate = valid & ~(same_class & same_signer)
    usable = positive.any(dim=1) & candidate.any(dim=1)
    coverage = positive.any(dim=1).to(logits.dtype).mean()
    if not usable.any():
        return zero, coverage

    neg_inf = logits.new_tensor(-1e30)
    candidate_logits = logits.masked_fill(~candidate, neg_inf)
    positive_logits = logits.masked_fill(~positive, neg_inf)
    log_denominator = torch.logsumexp(candidate_logits, dim=1)
    log_positive_mean = torch.logsumexp(
        positive_logits, dim=1
    ) - positive.sum(dim=1).clamp_min(1).log()
    return (log_denominator - log_positive_mean)[usable].mean(), coverage


class FDXMPModel(SLRModel):
    """VAC backbone plus a CTC-aligned class-prototype calibration channel."""

    def __init__(
        self,
        num_classes,
        c2d_type,
        conv_type,
        use_bn=False,
        hidden_size=1024,
        gloss_dict=None,
        loss_weights=None,
        weight_norm=True,
        share_classifier=True,
        beam_width=10,
        num_processes=10,
        rescorer_path=None,
        rescorer_alpha=0.0,
        rescorer_beta=0.0,
        proto_enabled=True,
        proto_momentum=0.9,
        proto_confidence=0.5,
        proto_temperature=0.1,
        proto_gate_init=0.05,
        proto_align_weight=0.01,
        proto_align_warmup_epochs=0,
        proto_mode="cosine",
        proto_variance_shrinkage=0.1,
        proto_variance_floor=1e-4,
        class_prior_path=None,
        prior_tau=0.0,
        fdgc_enabled=False,
        fdgc_tau_max=2.0,
        fdgc_delta_max=0.5,
        fdgc_blank_bias_max=1.0,
        signer_film=False,
        max_signer_id=31,
        signer_rank=4,
        signer_adversarial_strength=0.1,
        signer_inv_temperature=0.1,
        signer_inv_queue_size=2048,
        signer_inv_confidence=0.1,
        signer_inv_min_mass=0.05,
        text_proto_enabled=False,
        text_embedding_dim=256,
        text_ngram_min=3,
        text_ngram_max=5,
        text_temperature=0.1,
        text_gate_init=0.01,
        text_alignment_temperature=0.1,
        text_confidence=0.1,
        text_min_visual_support=0.05,
        text_align_warmup_epochs=0,
        text_embedding_path=None,
        text_embedding_source="lexical",
        multichannel_enabled=False,
        multichannel_min_temperature=0.25,
        multichannel_max_temperature=4.0,
        weighted_ctc_enabled=False,
        weighted_ctc_max=2.0,
        weighted_ctc_temperature=1.0,
        weighted_ctc_mapping="linear",
        weighted_ctc_blank_weight=1.0,
        pra_enabled=False,
        pra_max_strength=0.25,
        pra_min_mix_mass=0.05,
        pra_min_prototype_support=0.05,
        pra_confidence_threshold=0.0,
        rdb_enabled=False,
        rdb_max_strength=0.20,
        rdb_hidden_size=64,
        rdb_kernel_size=3,
        rdb_min_content_mass=0.05,
        rdb_min_prototype_support=0.05,
        rdb_confidence_threshold=0.1,
        rdb_recon_warmup_epochs=5,
        rdb_recon_mse_weight=1.0,
        rdb_recon_cosine_weight=0.1,
        rdb_recon_temperature=0.1,
        rdb_style_separation_margin=1e-4,
        rdb_style_separation_weight=1.0,
    ):
        active_losses = {
            str(name)
            for name, weight in (loss_weights or {}).items()
            if float(weight) != 0.0
        }
        module_requirements = {
            "WeightedCTC": weighted_ctc_enabled,
            "PRA": pra_enabled,
            "RDBRecon": rdb_enabled,
            "RDBCTC": rdb_enabled,
            "RDBStyle": rdb_enabled,
            "ProtoAlign": proto_enabled,
            "TextAlign": text_proto_enabled,
        }
        for loss_name, enabled in module_requirements.items():
            if loss_name in active_losses and not enabled:
                raise ValueError(
                    "{} loss requires its model module to be enabled".format(
                        loss_name
                    )
                )
        signer_losses = {
            "SignerReg",
            "SignerCE",
            "SignerAdv",
            "SignerInv",
            "SignerOrth",
        }
        if active_losses.intersection(signer_losses) and not signer_film:
            raise ValueError("signer loss requires signer_film=True")
        rdb_losses = {"RDBRecon", "RDBCTC", "RDBStyle"}
        if "RDBStyle" in active_losses and not signer_film:
            raise ValueError("RDBStyle loss requires signer_film=True")
        if pra_enabled and not proto_enabled:
            raise ValueError("PRA requires visual prototypes")
        if rdb_enabled and not proto_enabled:
            raise ValueError("RDB requires visual prototypes")
        if text_proto_enabled and not proto_enabled:
            raise ValueError("text prototype alignment requires visual prototypes")
        if weighted_ctc_enabled and "WeightedCTC" not in active_losses:
            raise ValueError(
                "weighted_ctc_enabled requires a non-zero WeightedCTC loss"
            )
        if pra_enabled and "PRA" not in active_losses:
            raise ValueError("pra_enabled requires a non-zero PRA loss")
        if rdb_enabled and not active_losses.intersection(rdb_losses):
            raise ValueError(
                "rdb_enabled requires at least one non-zero RDB loss"
            )
        if rdb_enabled and "RDBCTC" not in active_losses:
            raise ValueError(
                "rdb_enabled requires a non-zero RDBCTC loss "
                "to bootstrap the identity-initialized decoder"
            )
        super().__init__(
            num_classes=num_classes,
            c2d_type=c2d_type,
            conv_type=conv_type,
            use_bn=use_bn,
            hidden_size=hidden_size,
            gloss_dict=gloss_dict,
            loss_weights=loss_weights,
            weight_norm=weight_norm,
            share_classifier=share_classifier,
            beam_width=beam_width,
            num_processes=num_processes,
            rescorer_path=rescorer_path,
            rescorer_alpha=rescorer_alpha,
            rescorer_beta=rescorer_beta,
        )
        self.signer_film = bool(signer_film)
        self.supports_signer = self.signer_film
        if self.signer_film:
            if max_signer_id < 1:
                raise ValueError("max_signer_id must be positive")
            if signer_rank < 1:
                raise ValueError("signer_rank must be positive")
            self.signer_codes = nn.Embedding(
                max_signer_id + 1, signer_rank
            )
            nn.init.normal_(self.signer_codes.weight, std=0.1)
            self.signer_codes.weight.requires_grad_(False)
            self.register_buffer(
                "signer_seen",
                torch.zeros(max_signer_id + 1, dtype=torch.bool),
            )
            self.signer_gamma_basis = nn.Parameter(
                torch.zeros(signer_rank, hidden_size)
            )
            self.signer_beta_basis = nn.Parameter(
                torch.zeros(signer_rank, hidden_size)
            )
            self.signer_branch_projection = nn.Linear(
                hidden_size, hidden_size, bias=False
            )
            self.gloss_branch_projection = nn.Linear(
                hidden_size, hidden_size, bias=False
            )
            nn.init.eye_(self.signer_branch_projection.weight)
            nn.init.eye_(self.gloss_branch_projection.weight)
            split_size = hidden_size // 2
            split_mask = torch.zeros(hidden_size)
            split_mask[:split_size] = 1.0
            self.register_buffer(
                "signer_feature_mask", split_mask, persistent=False
            )
            self.signer_classifier = nn.Linear(
                hidden_size, max_signer_id + 1
            )
            self.gloss_signer_classifier = nn.Linear(
                hidden_size, max_signer_id + 1
            )
            self.signer_adversarial_strength = float(
                signer_adversarial_strength
            )
            self.signer_inv_temperature = float(signer_inv_temperature)
            self.signer_inv_confidence = float(signer_inv_confidence)
            self.signer_inv_min_mass = float(signer_inv_min_mass)
            if self.signer_inv_temperature <= 0.0:
                raise ValueError("signer_inv_temperature must be positive")
            if signer_inv_queue_size < 1:
                raise ValueError("signer_inv_queue_size must be positive")
            self.register_buffer(
                "signer_occurrence_embeddings",
                torch.zeros(int(signer_inv_queue_size), hidden_size),
                persistent=False,
            )
            self.register_buffer(
                "signer_occurrence_classes",
                torch.full((int(signer_inv_queue_size),), -1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "signer_occurrence_signers",
                torch.full((int(signer_inv_queue_size),), -1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "signer_occurrence_valid",
                torch.zeros(int(signer_inv_queue_size), dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "signer_occurrence_ptr",
                torch.zeros((), dtype=torch.long),
                persistent=False,
            )
        else:
            self.signer_codes = None
            self.register_parameter("signer_gamma_basis", None)
            self.register_parameter("signer_beta_basis", None)
            self.signer_branch_projection = None
            self.gloss_branch_projection = None
            self.signer_classifier = None
            self.gloss_signer_classifier = None
            self.signer_adversarial_strength = 0.0
            self.signer_inv_temperature = 0.0
            self.signer_inv_confidence = 0.0
            self.signer_inv_min_mass = 0.0
        self.proto_enabled = bool(proto_enabled)
        if self.proto_enabled:
            self.prototype_head = CTCPrototypeHead(
                num_classes=num_classes,
                feature_dim=hidden_size,
                momentum=proto_momentum,
                confidence=proto_confidence,
                temperature=proto_temperature,
                distribution=proto_mode,
                variance_shrinkage=proto_variance_shrinkage,
                variance_floor=proto_variance_floor,
            )
            gate_probability = min(
                max(float(proto_gate_init), 1e-4),
                1.0 - 1e-4,
            )
            self.prototype_gate = nn.Parameter(
                torch.tensor(
                    math.log(gate_probability / (1.0 - gate_probability))
                )
            )
        else:
            self.prototype_head = None
            self.register_parameter("prototype_gate", None)
        self.pra = (
            PrototypeRecombination(
                max_strength=pra_max_strength,
                min_mix_mass=pra_min_mix_mass,
                min_prototype_support=pra_min_prototype_support,
                confidence_threshold=pra_confidence_threshold,
                blank_id=0,
            )
            if pra_enabled
            else None
        )
        self.rdb_enabled = bool(rdb_enabled)
        self.rdb = (
            RDB(
                feature_dim=hidden_size,
                max_strength=rdb_max_strength,
                hidden_size=rdb_hidden_size,
                kernel_size=rdb_kernel_size,
                min_content_mass=rdb_min_content_mass,
                min_prototype_support=rdb_min_prototype_support,
                confidence_threshold=rdb_confidence_threshold,
                blank_id=0,
                style_separation_margin=rdb_style_separation_margin,
                reconstruction_mse_weight=rdb_recon_mse_weight,
                reconstruction_cosine_weight=rdb_recon_cosine_weight,
                reconstruction_temperature=rdb_recon_temperature,
            )
            if self.rdb_enabled
            else None
        )
        self.rdb_recon_warmup_epochs = int(rdb_recon_warmup_epochs)
        self.rdb_style_separation_weight = float(
            rdb_style_separation_weight
        )
        if self.rdb_recon_warmup_epochs < 0:
            raise ValueError("rdb_recon_warmup_epochs must be non-negative")
        if (
            not math.isfinite(self.rdb_style_separation_weight)
            or self.rdb_style_separation_weight < 0.0
        ):
            raise ValueError(
                "rdb_style_separation_weight must be finite and non-negative"
            )
        self.text_proto_enabled = bool(text_proto_enabled)
        if self.text_proto_enabled:
            self.text_prototype_head = TextPrototypeHead(
                num_classes=num_classes,
                feature_dim=hidden_size,
                gloss_dict=gloss_dict,
                embedding_dim=text_embedding_dim,
                ngram_min=text_ngram_min,
                ngram_max=text_ngram_max,
                temperature=text_temperature,
                gate_init=text_gate_init,
                alignment_temperature=text_alignment_temperature,
                confidence_threshold=text_confidence,
                min_visual_support=text_min_visual_support,
                text_embedding_path=text_embedding_path,
                embedding_source=text_embedding_source,
            )
        else:
            self.text_prototype_head = None
        self.multichannel_enabled = bool(multichannel_enabled)
        if self.multichannel_enabled and self.text_prototype_head is None:
            raise ValueError(
                "multichannel calibration requires the text prototype channel"
            )
        self.multichannel_calibrator = (
            MultiChannelCalibrator(
                num_classes=num_classes,
                min_temperature=multichannel_min_temperature,
                max_temperature=multichannel_max_temperature,
            )
            if self.multichannel_enabled
            else None
        )
        self.prototype_align_weight = float(proto_align_weight)
        self.proto_align_warmup_epochs = int(proto_align_warmup_epochs)
        self.text_align_warmup_epochs = int(text_align_warmup_epochs)
        if self.proto_align_warmup_epochs < 0:
            raise ValueError("proto_align_warmup_epochs must be non-negative")
        if self.text_align_warmup_epochs < 0:
            raise ValueError("text_align_warmup_epochs must be non-negative")
        # None preserves the legacy fixed-weight behavior until a trainer sets an epoch.
        self._training_epoch = None
        self.prior_tau = float(prior_tau)
        if class_prior_path:
            prior = np.load(class_prior_path).astype(np.float32)
            prior = np.clip(prior, 1e-8, None)
            prior = prior / prior.sum()
            self.register_buffer("class_log_prior", torch.from_numpy(np.log(prior)))
        else:
            self.register_buffer("class_log_prior", torch.zeros(num_classes))
        self.weighted_ctc_enabled = bool(weighted_ctc_enabled)
        if self.weighted_ctc_enabled:
            if class_prior_path is None:
                raise ValueError(
                    "weighted CTC requires class_prior_path for class counts"
                )
            ctc_weights = class_weights_from_prior(
                torch.from_numpy(prior),
                blank_weight=weighted_ctc_blank_weight,
                max_weight=weighted_ctc_max,
                temperature=weighted_ctc_temperature,
                mapping=weighted_ctc_mapping,
            )
            self.register_buffer("ctc_class_weights", ctc_weights)
        else:
            self.register_buffer(
                "ctc_class_weights",
                torch.ones(num_classes),
            )
        self.fdgc_enabled = bool(fdgc_enabled)
        self.fdgc = (
            FDGCCalibrator(
                num_classes=num_classes,
                tau_max=fdgc_tau_max,
                delta_max=fdgc_delta_max,
                blank_bias_max=fdgc_blank_bias_max,
                blank_id=0,
            )
            if self.fdgc_enabled
            else None
        )
        self.register_buffer(
            "fdgc_difficulty", torch.zeros(num_classes - 1)
        )

    def _prior_adjusted_logits(self, logits):
        if self.prior_tau <= 0.0:
            return logits
        adjusted = logits.clone()
        adjusted[:, :, 1:] = (
            adjusted[:, :, 1:]
            - self.prior_tau * self.class_log_prior[1:].view(1, 1, -1)
        )
        return adjusted

    def _nonblank_source_prior(self):
        prior = self.class_log_prior.exp()[1:]
        return prior / prior.sum().clamp_min(1e-8)

    def set_fdgc_difficulty(self, difficulty):
        difficulty = torch.as_tensor(
            difficulty,
            dtype=self.fdgc_difficulty.dtype,
            device=self.fdgc_difficulty.device,
        )
        if difficulty.numel() != self.fdgc_difficulty.numel():
            raise ValueError("difficulty must have one value per non-blank class")
        self.fdgc_difficulty.copy_(difficulty.clamp(0.0, 1.0))

    def set_training_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("training epoch must be non-negative")
        self._training_epoch = epoch

    def _alignment_warmup_factor(self, loss_name):
        if self._training_epoch is None:
            return 1.0
        if loss_name == "ProtoAlign":
            warmup_epochs = self.proto_align_warmup_epochs
        elif loss_name == "TextAlign":
            warmup_epochs = self.text_align_warmup_epochs
        elif loss_name == "RDBRecon":
            warmup_epochs = self.rdb_recon_warmup_epochs
        else:
            return 1.0
        if warmup_epochs <= 0:
            return 1.0
        return min(float(self._training_epoch) / warmup_epochs, 1.0)

    @staticmethod
    def _length_masked_mean(features, lengths):
        max_length = features.size(0)
        frame_mask = (
            torch.arange(max_length, device=features.device).view(1, -1)
            < lengths.to(features.device).view(-1, 1)
        )
        mask = frame_mask.transpose(0, 1).unsqueeze(-1)
        total = features.masked_fill(~mask, 0.0).sum(dim=0)
        return total / lengths.to(features.device).clamp_min(1).view(-1, 1)

    def _collect_signer_occurrences(
        self,
        features,
        gamma,
        confidence,
        signer_ids,
        label,
        label_lgt,
        feat_len,
    ):
        embeddings = []
        classes = []
        signers = []
        offset = 0
        lengths = label_lgt.detach().cpu().tolist()
        feature_lengths = feat_len.detach().cpu().tolist()
        for batch_idx, label_length in enumerate(lengths):
            target = label[offset : offset + label_length]
            offset += label_length
            sequence_length = min(
                int(feature_lengths[batch_idx]), features.size(0)
            )
            if sequence_length <= 0 or target.numel() == 0:
                continue
            for class_id in target.detach().unique():
                class_id = int(class_id.item())
                if class_id <= 0 or class_id >= gamma.size(-1):
                    continue
                frame_confidence = confidence[:sequence_length, batch_idx]
                weights = gamma[:sequence_length, batch_idx, class_id]
                weights = weights * (
                    frame_confidence >= self.signer_inv_confidence
                ).to(weights.dtype)
                mass = weights.sum()
                if mass < self.signer_inv_min_mass:
                    continue
                embedding = (
                    features[:sequence_length, batch_idx]
                    * weights.unsqueeze(-1)
                ).sum(dim=0) / mass.clamp_min(1e-8)
                embeddings.append(embedding)
                classes.append(class_id)
                signers.append(int(signer_ids[batch_idx].item()))
        if not embeddings:
            return None
        return (
            torch.stack(embeddings, dim=0),
            torch.tensor(classes, dtype=torch.long, device=features.device),
            torch.tensor(signers, dtype=torch.long, device=features.device),
        )

    @torch.no_grad()
    def _update_signer_occurrence_memory(
        self, embeddings, classes, signers
    ):
        count = embeddings.size(0)
        capacity = self.signer_occurrence_embeddings.size(0)
        indices = (
            self.signer_occurrence_ptr
            + torch.arange(count, device=embeddings.device)
        ) % capacity
        self.signer_occurrence_embeddings[indices] = embeddings.detach()
        self.signer_occurrence_classes[indices] = classes
        self.signer_occurrence_signers[indices] = signers
        self.signer_occurrence_valid[indices] = True
        next_ptr = (int(self.signer_occurrence_ptr.item()) + count) % capacity
        self.signer_occurrence_ptr.fill_(next_ptr)

    def _signer_probe_logits(self, signer_summary, gloss_summary):
        signer_logits = self.signer_classifier(signer_summary)
        reversed_gloss_summary = _GradientReversal.apply(
            gloss_summary, self.signer_adversarial_strength
        )
        gloss_signer_logits = self.gloss_signer_classifier(
            reversed_gloss_summary
        )
        return signer_logits, gloss_signer_logits

    def _signer_style_context(self, signer_ids):
        codes = self.signer_codes(signer_ids.long())
        codes = codes * self.signer_seen[signer_ids.long()].unsqueeze(-1).to(
            codes.dtype
        )
        gamma = codes @ self.signer_gamma_basis
        beta = codes @ self.signer_beta_basis
        return torch.cat((gamma, beta), dim=-1)

    def _counterfactual_signer_ids(self, signer_ids):
        signer_ids = signer_ids.detach().long()
        alternate_ids = signer_ids.clone()
        swap_mask = torch.zeros(
            signer_ids.numel(),
            dtype=torch.bool,
            device=signer_ids.device,
        )
        seen_ids = torch.nonzero(
            self.signer_seen, as_tuple=False
        ).flatten().to(signer_ids.device)
        if seen_ids.numel() < 2:
            return alternate_ids, swap_mask
        for batch_idx, signer_id in enumerate(signer_ids.tolist()):
            candidates = seen_ids[seen_ids != int(signer_id)]
            if candidates.numel() == 0:
                continue
            candidate_idx = (
                batch_idx + int(signer_id)
            ) % int(candidates.numel())
            alternate_ids[batch_idx] = candidates[candidate_idx]
            swap_mask[batch_idx] = True
        return alternate_ids, swap_mask

    @staticmethod
    def _select_label_sequences(label, label_lgt, selection):
        selection = selection.detach().cpu().bool().tolist()
        lengths = label_lgt.detach().cpu().long().tolist()
        pieces = []
        selected_lengths = []
        offset = 0
        for selected, length in zip(selection, lengths):
            target = label[offset : offset + length]
            offset += length
            if selected:
                pieces.append(target)
                selected_lengths.append(length)
        if pieces:
            selected_label = torch.cat(pieces)
        else:
            selected_label = label.new_empty((0,))
        selected_lgt = label_lgt.new_tensor(selected_lengths, dtype=torch.long)
        return selected_label, selected_lgt

    def forward(
        self,
        x,
        len_x,
        label=None,
        label_lgt=None,
        decode=True,
        signer_ids=None,
    ):
        if len(x.shape) == 5:
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        visual_feat = conv1d_outputs["visual_feat"]
        feat_len = conv1d_outputs["feat_len"]
        temporal_outputs = self.temporal_model(visual_feat, feat_len)
        temporal_features = temporal_outputs["predictions"]
        conditioned_features = temporal_features
        signer_branch_features = None
        gloss_branch_features = None
        conditioned_gloss_features = None
        signer_gamma = None
        signer_beta = None
        signer_regularization = conditioned_features.sum() * 0.0
        signer_logits = None
        gloss_signer_logits = None
        signer_loss = conditioned_features.sum() * 0.0
        gloss_signer_loss = conditioned_features.sum() * 0.0
        signer_inv_loss = conditioned_features.sum() * 0.0
        signer_inv_coverage = conditioned_features.new_tensor(0.0)
        signer_orthogonal_loss = conditioned_features.sum() * 0.0
        if self.signer_film:
            if signer_ids is None:
                raise ValueError("signer_ids are required when signer_film is enabled")
            signer_ids = signer_ids.to(conditioned_features.device)
            if self.training:
                self.signer_seen[signer_ids] = True
            signer_mask = self.signer_feature_mask.to(
                temporal_features.device
            )
            signer_branch_features = self.signer_branch_projection(
                temporal_features * signer_mask
            )
            gloss_branch_features = self.gloss_branch_projection(
                temporal_features * (1.0 - signer_mask)
            )
            signer_summary = self._length_masked_mean(
                signer_branch_features, feat_len
            )
            codes = self.signer_codes(signer_ids)
            codes = codes * self.signer_seen[signer_ids].unsqueeze(-1).to(
                codes.dtype
            )
            signer_gamma = codes @ self.signer_gamma_basis
            signer_beta = codes @ self.signer_beta_basis
            conditioned_gloss_features = (
                (1.0 + signer_gamma.unsqueeze(0))
                * gloss_branch_features
                + signer_beta.unsqueeze(0)
            )
            conditioned_features = (
                signer_branch_features + conditioned_gloss_features
            )
            signer_regularization = (
                signer_gamma.square().mean() + signer_beta.square().mean()
            )
            gloss_summary = self._length_masked_mean(
                conditioned_gloss_features, feat_len
            )
            signer_logits, gloss_signer_logits = self._signer_probe_logits(
                signer_summary,
                gloss_summary,
            )
            signer_loss = F.cross_entropy(
                signer_logits, signer_ids.long()
            )
            gloss_signer_loss = F.cross_entropy(
                gloss_signer_logits, signer_ids.long()
            )
            signer_cosine = F.cosine_similarity(
                signer_summary, gloss_summary, dim=-1
            )
            signer_orthogonal_loss = F.relu(signer_cosine).square().mean()
        base_logits = self.classifier(conditioned_features)

        gamma = None
        confidence = None
        proto_align_loss = base_logits.sum() * 0.0
        if self.training and label is not None and label_lgt is not None:
            gamma, confidence = collect_batch_posteriors(
                base_logits,
                label,
                label_lgt.cpu(),
                feature_lengths=feat_len.detach().cpu(),
                blank_id=0,
            )
            if self.prototype_head is not None:
                self.prototype_head.update(
                    conditioned_features.detach(),
                    gamma.detach(),
                    confidence.detach(),
                )
            if self.signer_film:
                occurrences = self._collect_signer_occurrences(
                    conditioned_gloss_features,
                    gamma,
                    confidence,
                    signer_ids,
                    label,
                    label_lgt,
                    feat_len,
                )
                if occurrences is not None:
                    occurrence_embeddings, occurrence_classes, occurrence_signers = (
                        occurrences
                    )
                    signer_inv_loss, signer_inv_coverage = (
                        signer_multi_positive_loss(
                            occurrence_embeddings,
                            occurrence_classes,
                            occurrence_signers,
                            self.signer_occurrence_embeddings,
                            self.signer_occurrence_classes,
                            self.signer_occurrence_signers,
                            self.signer_occurrence_valid,
                            self.signer_inv_temperature,
                        )
                    )
                    self._update_signer_occurrence_memory(
                        occurrence_embeddings.detach(),
                        occurrence_classes,
                        occurrence_signers,
                    )

        prototype_logits = None
        gate = None
        fused_logits = base_logits
        if self.prototype_head is not None:
            prototype_logits = self.prototype_head(conditioned_features)
            gate = torch.sigmoid(self.prototype_gate)
            fused_logits = base_logits + gate * prototype_logits
        rdb_features = None
        rdb_logits = None
        rdb_counterfactual_features = None
        rdb_style_logits = None
        rdb_style_feat_len = None
        rdb_style_swap_mask = None
        rdb_recon_loss = None
        rdb_recon_mse = None
        rdb_recon_cosine = None
        rdb_style_separation_loss = None
        rdb_strength = None
        rdb_active_fraction = None
        rdb_mean_shift = None
        rdb_style_shift = None
        if self.rdb is not None and gamma is not None:
            style = None
            counterfactual_style = None
            style_swap_mask = None
            if self.signer_film:
                style = torch.cat((signer_gamma, signer_beta), dim=-1)
                alternate_signers, style_swap_mask = (
                    self._counterfactual_signer_ids(signer_ids)
                )
                if style_swap_mask.any():
                    counterfactual_style = self._signer_style_context(
                        alternate_signers
                    )
            rdb_output = self.rdb(
                conditioned_features,
                self.prototype_head.prototypes,
                self.prototype_head.support,
                gamma,
                confidence,
                lengths=feat_len,
                style=style,
                counterfactual_style=counterfactual_style,
                style_swap_mask=style_swap_mask,
            )
            rdb_features = rdb_output["features"]
            rdb_counterfactual_features = rdb_output[
                "counterfactual_features"
            ]
            rdb_logits = self.classifier(rdb_features)
            reconstruction = self.rdb.reconstruction_loss(
                conditioned_features,
                rdb_features,
                lengths=feat_len,
                active=rdb_output["active"],
            )
            rdb_recon_loss = reconstruction["loss"]
            rdb_recon_mse = reconstruction["mse"]
            rdb_recon_cosine = reconstruction["cosine"]
            rdb_style_separation_loss = rdb_output[
                "style_separation_loss"
            ]
            rdb_strength = rdb_output["strength"]
            rdb_active_fraction = rdb_output["active"].to(
                conditioned_features.dtype
            ).mean()
            rdb_mean_shift = rdb_output["mean_shift"]
            rdb_style_shift = rdb_output["style_shift"]
            if (
                rdb_counterfactual_features is not None
                and style_swap_mask is not None
                and style_swap_mask.any()
            ):
                rdb_style_swap_mask = style_swap_mask
                style_length_mask = style_swap_mask.to(feat_len.device)
                rdb_style_feat_len = feat_len[style_length_mask]
                rdb_style_logits = self.classifier(
                    rdb_counterfactual_features[
                        :,
                        style_swap_mask.to(conditioned_features.device),
                    ]
                )
        pra_logits = None
        pra_features = None
        pra_strength = None
        pra_active_fraction = None
        if self.pra is not None and gamma is not None:
            pra_output = self.pra(
                conditioned_features,
                self.prototype_head.prototypes,
                self.prototype_head.support,
                gamma,
                confidence,
            )
            pra_features = pra_output["features"]
            pra_logits = self.classifier(pra_features)
            pra_strength = pra_output["strength"]
            pra_active_fraction = pra_output["active"].to(
                conditioned_features.dtype
            ).mean()
        text_logits = None
        text_gate = None
        text_align_loss = conditioned_features.sum() * 0.0
        if self.text_prototype_head is not None:
            text_logits = self.text_prototype_head(conditioned_features)
            text_gate = self.text_prototype_head.gate()
            fused_logits = fused_logits + text_gate * text_logits
        if gamma is not None and self.prototype_head is not None:
            proto_align_loss = self.prototype_head.alignment_loss(
                prototype_logits, gamma, confidence
            )
            if self.text_prototype_head is not None:
                text_align_loss = self.text_prototype_head.alignment_loss(
                    text_logits,
                    gamma,
                    confidence,
                    self.prototype_head.prototypes,
                    self.prototype_head.support,
                )

        multichannel_probabilities = None
        multichannel_temperatures = None
        multichannel_weights = None
        sequence_log_probs = None
        if self.multichannel_calibrator is not None:
            multichannel_output = self.multichannel_calibrator(
                self._prior_adjusted_logits(base_logits),
                prototype_logits,
                text_logits,
            )
            fused_logits = multichannel_output["log_probs"]
            sequence_log_probs = multichannel_output["log_probs"]
            multichannel_probabilities = multichannel_output["probabilities"]
            multichannel_temperatures = multichannel_output["temperatures"]
            multichannel_weights = multichannel_output["weights"]

        recognized_sents = None
        conv_sents = None
        fdgc_logits = None
        fdgc_log_probs = None
        fdgc_tau = None
        fdgc_blank_bias = None
        if not self.training and decode:
            if self.multichannel_calibrator is None:
                decode_logits = self._prior_adjusted_logits(fused_logits)
            else:
                decode_logits = fused_logits
            if self.fdgc_enabled:
                fdgc_output = self.fdgc(
                    decode_logits,
                    source_prior=self._nonblank_source_prior(),
                    difficulty=self.fdgc_difficulty,
                    input_is_log_probs=(
                        self.multichannel_calibrator is not None
                    ),
                )
                decode_logits = fdgc_output["logits"]
                fdgc_logits = fdgc_output["logits"]
                fdgc_log_probs = fdgc_output["log_probs"]
                fdgc_tau = fdgc_output["tau"]
                fdgc_blank_bias = fdgc_output["blank_bias"]
            decode_conv = self._prior_adjusted_logits(conv1d_outputs["conv_logits"])
            recognized_sents = self.decoder.decode(
                (
                    multichannel_probabilities
                    if self.multichannel_calibrator is not None
                    and not self.fdgc_enabled
                    else decode_logits
                ),
                feat_len,
                batch_first=False,
                probs=(
                    self.multichannel_calibrator is not None
                    and not self.fdgc_enabled
                ),
            )
            conv_sents = self.decoder.decode(
                decode_conv, feat_len, batch_first=False, probs=False
            )

        return {
            "framewise_features": framewise,
            "visual_features": visual_feat,
            "temporal_features": temporal_features,
            "conditioned_features": conditioned_features,
            "signer_branch_features": signer_branch_features,
            "gloss_branch_features": gloss_branch_features,
            "conditioned_gloss_features": conditioned_gloss_features,
            "feat_len": feat_len,
            "conv_logits": conv1d_outputs["conv_logits"],
            "base_logits": base_logits,
            "prototype_logits": prototype_logits,
            "pra_features": pra_features,
            "pra_logits": pra_logits,
            "pra_strength": pra_strength,
            "pra_active_fraction": pra_active_fraction,
            "rdb_features": rdb_features,
            "rdb_logits": rdb_logits,
            "rdb_counterfactual_features": rdb_counterfactual_features,
            "rdb_style_logits": rdb_style_logits,
            "rdb_style_feat_len": rdb_style_feat_len,
            "rdb_style_swap_mask": rdb_style_swap_mask,
            "rdb_recon_loss": rdb_recon_loss,
            "rdb_recon_mse": rdb_recon_mse,
            "rdb_recon_cosine": rdb_recon_cosine,
            "rdb_style_separation_loss": rdb_style_separation_loss,
            "rdb_strength": rdb_strength,
            "rdb_active_fraction": rdb_active_fraction,
            "rdb_mean_shift": rdb_mean_shift,
            "rdb_style_shift": rdb_style_shift,
            "text_logits": text_logits,
            "text_gate": text_gate,
            "sequence_logits": fused_logits,
            "sequence_log_probs": sequence_log_probs,
            "multichannel_probabilities": multichannel_probabilities,
            "multichannel_temperatures": multichannel_temperatures,
            "multichannel_weights": multichannel_weights,
            "prototype_gate": gate,
            "signer_gamma": signer_gamma,
            "signer_beta": signer_beta,
            "signer_regularization": signer_regularization,
            "signer_logits": signer_logits,
            "signer_loss": signer_loss,
            "gloss_signer_logits": gloss_signer_logits,
            "gloss_signer_loss": gloss_signer_loss,
            "signer_inv_loss": signer_inv_loss,
            "signer_inv_coverage": signer_inv_coverage,
            "signer_orthogonal_loss": signer_orthogonal_loss,
            "proto_align_loss": proto_align_loss,
            "text_align_loss": text_align_loss,
            "fdgc_logits": fdgc_logits,
            "fdgc_log_probs": fdgc_log_probs,
            "fdgc_tau": fdgc_tau,
            "fdgc_blank_bias": fdgc_blank_bias,
            "conv_sents": conv_sents,
            "recognized_sents": recognized_sents,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = ret_dict["sequence_logits"].sum() * 0.0
        self.last_ctc_validity = {}
        self.last_rdb_style_available = False
        for name, weight in self.loss_weights.items():
            if name == "SeqCTC":
                loss = loss + weight * self._ctc_branch_loss(
                    ret_dict["sequence_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                    "SeqCTC",
                    log_probs=ret_dict.get("sequence_log_probs"),
                )
            elif name == "ConvCTC":
                loss = loss + weight * self._ctc_branch_loss(
                    ret_dict["conv_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                    "ConvCTC",
                )
            elif name == "WeightedCTC":
                loss = loss + weight * self._ctc_branch_loss(
                    ret_dict["base_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                    "WeightedCTC",
                    class_weights=self.ctc_class_weights,
                )
            elif name == "Dist":
                if ret_dict.get("sequence_log_probs") is None:
                    distillation_loss = self.loss["distillation"](
                        ret_dict["conv_logits"],
                        ret_dict["sequence_logits"].detach(),
                        use_blank=False,
                    )
                else:
                    distillation_loss = self.loss["distillation"](
                        ret_dict["conv_logits"],
                        ret_dict["sequence_log_probs"].detach(),
                        use_blank=False,
                        ref_is_log_probs=True,
                    )
                loss = loss + weight * distillation_loss
            elif name == "ProtoAlign":
                scale = self._alignment_warmup_factor(name)
                loss = loss + weight * scale * ret_dict["proto_align_loss"]
            elif name == "TextAlign":
                scale = self._alignment_warmup_factor(name)
                loss = loss + weight * scale * ret_dict["text_align_loss"]
            elif name == "PRA":
                if ret_dict.get("pra_logits") is None:
                    raise ValueError("PRA loss requires training posteriors")
                loss = loss + weight * self._ctc_branch_loss(
                    ret_dict["pra_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                    "PRA",
                )
            elif name == "RDBRecon":
                if ret_dict.get("rdb_recon_loss") is None:
                    raise ValueError(
                        "RDBRecon loss requires training posteriors"
                    )
                scale = self._alignment_warmup_factor(name)
                loss = loss + weight * scale * ret_dict["rdb_recon_loss"]
            elif name == "RDBCTC":
                if ret_dict.get("rdb_logits") is None:
                    raise ValueError(
                        "RDBCTC loss requires reconstructed logits"
                    )
                loss = loss + weight * self._ctc_branch_loss(
                    ret_dict["rdb_logits"],
                    label,
                    ret_dict["feat_len"],
                    label_lgt,
                    "RDBCTC",
                )
            elif name == "RDBStyle":
                rdb_style_logits = ret_dict.get("rdb_style_logits")
                style_swap_mask = ret_dict.get("rdb_style_swap_mask")
                if rdb_style_logits is None or style_swap_mask is None:
                    self.last_rdb_style_available = False
                    continue
                style_label, style_label_lgt = self._select_label_sequences(
                    label,
                    label_lgt,
                    style_swap_mask,
                )
                if style_label_lgt.numel() == 0:
                    self.last_rdb_style_available = False
                    continue
                style_loss = self._ctc_branch_loss(
                    rdb_style_logits,
                    style_label,
                    ret_dict["rdb_style_feat_len"],
                    style_label_lgt,
                    "RDBStyle",
                )
                style_loss = (
                    style_loss
                    + self.rdb_style_separation_weight
                    * ret_dict["rdb_style_separation_loss"]
                )
                loss = loss + weight * style_loss
                self.last_rdb_style_available = True
            elif name == "SignerReg":
                loss = loss + weight * ret_dict["signer_regularization"]
            elif name == "SignerCE":
                loss = loss + weight * ret_dict["signer_loss"]
            elif name == "SignerAdv":
                loss = loss + weight * ret_dict["gloss_signer_loss"]
            elif name == "SignerInv":
                loss = loss + weight * ret_dict["signer_inv_loss"]
            elif name == "SignerOrth":
                loss = loss + weight * ret_dict["signer_orthogonal_loss"]
        return loss
