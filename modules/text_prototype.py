import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize_gloss_text(value):
    """Canonicalize a gloss label without language-specific assumptions."""
    return " ".join(str(value).strip().lower().split())


def _lexical_features(text, output_dim, ngram_min, ngram_max):
    if output_dim < 2:
        raise ValueError("text embedding dimension must be at least two")
    if ngram_min < 1 or ngram_max < ngram_min:
        raise ValueError("invalid character n-gram range")

    features = torch.zeros(output_dim, dtype=torch.float32)
    grams = []
    tokens = _TOKEN_PATTERN.findall(normalize_gloss_text(text))
    for token in tokens:
        grams.append("word:" + token)
        bounded = "^" + token + "$"
        for width in range(ngram_min, min(ngram_max, len(bounded)) + 1):
            for start in range(len(bounded) - width + 1):
                grams.append(bounded[start:start + width])
    if not grams:
        grams.append("empty")

    scale = 1.0 / math.sqrt(len(grams))
    for gram in grams:
        digest = hashlib.blake2b(
            gram.encode("utf-8"), digest_size=16
        ).digest()
        index = int.from_bytes(digest[:8], "little") % output_dim
        sign = 1.0 if digest[8] & 1 else -1.0
        features[index] += sign * scale
    return F.normalize(features, dim=0, eps=1e-8)


def build_lexical_embeddings(
    gloss_dict,
    num_classes,
    output_dim,
    ngram_min=3,
    ngram_max=5,
):
    """Build deterministic label-only features ordered by gloss class id."""
    embeddings = torch.zeros(num_classes, output_dim, dtype=torch.float32)
    seen = set()
    for gloss, value in gloss_dict.items():
        class_id = int(value[0])
        if class_id < 1 or class_id >= num_classes:
            raise ValueError("gloss id is outside the non-blank class range")
        if class_id in seen:
            raise ValueError("duplicate gloss class id: {}".format(class_id))
        seen.add(class_id)
        embeddings[class_id] = _lexical_features(
            gloss,
            output_dim,
            ngram_min,
            ngram_max,
        )
    if len(seen) != num_classes - 1:
        raise ValueError(
            "gloss dictionary does not cover every non-blank class"
        )
    return embeddings


def _validate_text_embeddings(embeddings, num_classes, gloss_dict=None):
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    if embeddings.ndim != 2:
        raise ValueError("text embeddings must have shape [C,D]")
    if embeddings.size(0) != num_classes:
        raise ValueError(
            "text embeddings must cover every class including blank"
        )
    if embeddings.size(1) < 1:
        raise ValueError("text embedding dimension must be positive")
    if not torch.isfinite(embeddings).all():
        raise ValueError("text embeddings must be finite")
    if torch.count_nonzero(embeddings[0]).item() != 0:
        raise ValueError("the blank text embedding row must be zero")
    if gloss_dict is not None:
        expected = {
            int(value[0])
            for value in gloss_dict.values()
            if 0 < int(value[0]) < num_classes
        }
        missing = sorted(set(range(1, num_classes)).difference(expected))
        if missing:
            raise ValueError(
                "gloss dictionary is missing class ids: {}".format(missing[:8])
            )
    return embeddings


def load_text_embeddings(path, gloss_dict, num_classes, embedding_dim=None):
    """Load a class-ordered dense matrix from npy, npz or JSON."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("text embedding file does not exist: {}".format(path))
    suffix = path.suffix.lower()
    if suffix == ".npy":
        embeddings = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            if "embeddings" not in source.files:
                raise ValueError("npz text embeddings require an 'embeddings' array")
            embeddings = source["embeddings"]
    elif suffix == ".json":
        with open(path, encoding="utf-8") as source:
            payload = json.load(source)
        if isinstance(payload, dict) and "embeddings" in payload:
            embeddings = payload["embeddings"]
        elif isinstance(payload, dict):
            if embedding_dim is None:
                first = next(
                    (value for key, value in payload.items() if key != "<blank>"),
                    None,
                )
                if first is None:
                    raise ValueError("JSON embedding mapping is empty")
                embedding_dim = len(first)
            embeddings = np.zeros((num_classes, int(embedding_dim)), dtype=np.float32)
            for gloss, value in gloss_dict.items():
                vector = payload.get(gloss)
                if vector is None:
                    vector = payload.get(normalize_gloss_text(gloss))
                if vector is None:
                    continue
                class_id = int(value[0])
                embeddings[class_id] = np.asarray(vector, dtype=np.float32)
        else:
            embeddings = payload
    else:
        raise ValueError("text embeddings must use .npy, .npz or .json")
    embeddings = _validate_text_embeddings(
        embeddings,
        int(num_classes),
        gloss_dict=gloss_dict,
    )
    if embedding_dim is not None and embeddings.size(1) != int(embedding_dim):
        raise ValueError(
            "text embedding dimension {} does not match requested {}".format(
                embeddings.size(1),
                embedding_dim,
            )
        )
    return embeddings


class TextPrototypeHead(nn.Module):
    """Frozen lexical class anchors with a trainable projection."""

    def __init__(
        self,
        num_classes,
        feature_dim,
        gloss_dict,
        embedding_dim=256,
        ngram_min=3,
        ngram_max=5,
        temperature=0.1,
        gate_init=0.01,
        alignment_temperature=0.1,
        confidence_threshold=0.1,
        min_visual_support=0.05,
        text_embeddings=None,
        text_embedding_path=None,
        embedding_source="lexical",
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include blank and one gloss")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if alignment_temperature <= 0.0:
            raise ValueError("alignment_temperature must be positive")
        allowed_sources = {
            "lexical",
            "wordnet+lexical",
            "llama",
            "wordnet+llama",
            "precomputed",
        }
        embedding_source = str(embedding_source)
        if embedding_source not in allowed_sources:
            raise ValueError(
                "embedding_source must be one of {}".format(
                    sorted(allowed_sources)
                )
            )
        if (
            embedding_source != "lexical"
            and text_embeddings is None
            and text_embedding_path is None
        ):
            raise ValueError(
                "semantic embedding_source requires precomputed embeddings"
            )
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.alignment_temperature = float(alignment_temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.min_visual_support = float(min_visual_support)
        if text_embeddings is not None and text_embedding_path is not None:
            raise ValueError(
                "provide either text_embeddings or text_embedding_path, not both"
            )
        if text_embedding_path is not None:
            lexical = load_text_embeddings(
                text_embedding_path,
                gloss_dict,
                self.num_classes,
                embedding_dim=embedding_dim,
            )
            self.embedding_source = embedding_source
            self.text_embedding_path = str(Path(text_embedding_path).resolve())
        elif text_embeddings is not None:
            lexical = _validate_text_embeddings(
                text_embeddings,
                self.num_classes,
                gloss_dict=gloss_dict,
            )
            self.embedding_source = embedding_source
            self.text_embedding_path = None
        else:
            lexical = build_lexical_embeddings(
                gloss_dict,
                self.num_classes,
                output_dim=embedding_dim,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
            )
            self.embedding_source = "lexical"
            self.text_embedding_path = None
        self.register_buffer("lexical_embeddings", lexical)
        self.projection = nn.Linear(
            lexical.size(1), self.feature_dim, bias=False
        )
        nn.init.xavier_uniform_(self.projection.weight)
        self.blank_bias = nn.Parameter(torch.zeros(()))
        gate_probability = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.gate_logit = nn.Parameter(
            torch.tensor(
                math.log(gate_probability / (1.0 - gate_probability))
            )
        )

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def projected_prototypes(self):
        return F.normalize(
            self.projection(self.lexical_embeddings[1:]),
            dim=-1,
            eps=1e-6,
        )

    def forward(self, features):
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                "feature dimension does not match text prototype head"
            )
        normalized = F.normalize(features, dim=-1, eps=1e-6)
        semantic_logits = (
            torch.matmul(normalized, self.projected_prototypes().t())
            / self.temperature
        )
        blank = self.blank_bias.to(semantic_logits.dtype).expand(
            *semantic_logits.shape[:-1], 1
        )
        return torch.cat([blank, semantic_logits], dim=-1)

    def alignment_loss(
        self,
        text_logits,
        gamma,
        confidence,
        visual_prototypes,
        visual_support,
    ):
        if text_logits.shape != gamma.shape:
            raise ValueError("text logits and CTC occupancy must match")
        confidence_mask = confidence >= self.confidence_threshold
        if not confidence_mask.any():
            return text_logits.sum() * 0.0

        normalized_gamma = gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        log_probs = text_logits.log_softmax(dim=-1)
        visual_to_text = -(
            normalized_gamma[confidence_mask]
            * log_probs[confidence_mask]
        ).sum(dim=-1).mean()

        valid_visual = (
            visual_support >= self.min_visual_support
        ).reshape(-1)
        valid_visual[0] = False
        visual_ids = valid_visual.nonzero(as_tuple=False).reshape(-1)
        if visual_ids.numel() < 2:
            return visual_to_text

        text_vectors = self.projected_prototypes()
        visual_vectors = F.normalize(
            visual_prototypes[visual_ids], dim=-1, eps=1e-6
        )
        similarity = torch.matmul(text_vectors, visual_vectors.t())
        similarity = similarity / self.alignment_temperature
        target = torch.full(
            (self.num_classes - 1,),
            -100,
            dtype=torch.long,
            device=similarity.device,
        )
        local_index = torch.full(
            (self.num_classes,),
            -1,
            dtype=torch.long,
            device=similarity.device,
        )
        local_index[visual_ids] = torch.arange(
            visual_ids.numel(), device=similarity.device
        )
        semantic_ids = torch.arange(
            1, self.num_classes, device=similarity.device
        )
        target[:] = local_index[semantic_ids]
        valid_text = target >= 0
        text_to_visual = F.cross_entropy(
            similarity[valid_text],
            target[valid_text],
        )
        return visual_to_text + text_to_visual
