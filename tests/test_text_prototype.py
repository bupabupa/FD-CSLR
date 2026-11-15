import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.text_prototype import (
    TextPrototypeHead,
    build_lexical_embeddings,
    load_text_embeddings,
)


def test_text_prototype_shapes_and_alignment():
    gloss_dict = {
        "essen": [1, 0],
        "trinken": [2, 1],
        "wetter": [3, 2],
        "regen": [4, 3],
        "sonne": [5, 4],
    }
    num_classes = len(gloss_dict) + 1
    first = build_lexical_embeddings(
        gloss_dict,
        num_classes=num_classes,
        output_dim=64,
    )
    second = build_lexical_embeddings(
        gloss_dict,
        num_classes=num_classes,
        output_dim=64,
    )
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        first[1:].norm(dim=-1),
        torch.ones(num_classes - 1),
    )
    assert torch.count_nonzero(first[0]) == 0

    head = TextPrototypeHead(
        num_classes=num_classes,
        feature_dim=32,
        gloss_dict=gloss_dict,
        embedding_dim=64,
        temperature=0.1,
        gate_init=0.01,
    )
    features = torch.randn(7, 2, 32)
    text_logits = head(features)
    assert text_logits.shape == (7, 2, num_classes)
    assert torch.isfinite(text_logits).all()
    assert head.gate().item() < 0.02

    gamma = torch.zeros(7, 2, num_classes)
    gamma[:, :, 1] = 0.8
    gamma[:, :, 0] = 0.2
    confidence = torch.ones(7, 2)
    visual_prototypes = torch.randn(num_classes, 32)
    visual_support = torch.full((num_classes,), 2.0)
    loss = head.alignment_loss(
        text_logits,
        gamma,
        confidence,
        visual_prototypes,
        visual_support,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert head.projection.weight.grad is not None
    assert head.projection.weight.grad.abs().sum().item() > 0


def test_text_alignment_skips_low_confidence_and_sparse_visuals():
    gloss_dict = {"a": [1, 0], "b": [2, 1], "c": [3, 2]}
    num_classes = 4
    head = TextPrototypeHead(
        num_classes=num_classes,
        feature_dim=8,
        gloss_dict=gloss_dict,
        embedding_dim=32,
        confidence_threshold=0.5,
    )
    features = torch.randn(3, 1, 8)
    text_logits = head(features)
    gamma = torch.zeros(3, 1, num_classes)
    gamma[:, :, 1] = 1.0
    confidence = torch.zeros(3, 1)
    visual_prototypes = torch.zeros(num_classes, 8)
    visual_support = torch.zeros(num_classes)
    empty_loss = head.alignment_loss(
        text_logits,
        gamma,
        confidence,
        visual_prototypes,
        visual_support,
    )
    assert empty_loss.item() == 0.0

    confidence[:] = 1.0
    only_visual_to_text = head.alignment_loss(
        text_logits,
        gamma,
        confidence,
        visual_prototypes,
        visual_support,
    )
    assert torch.isfinite(only_visual_to_text)
    assert only_visual_to_text.item() > 0.0


def test_precomputed_semantic_embeddings_are_loaded_with_provenance():
    gloss_dict = {"a": [1, 0], "b": [2, 1], "c": [3, 2]}
    num_classes = 4
    vectors = np.zeros((num_classes, 6), dtype=np.float32)
    vectors[1] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vectors[2] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    vectors[3] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "semantic.npy"
        np.save(path, vectors)
        loaded = load_text_embeddings(path, gloss_dict, num_classes)
        torch.testing.assert_close(loaded, torch.from_numpy(vectors))
        head = TextPrototypeHead(
            num_classes=num_classes,
            feature_dim=8,
            gloss_dict=gloss_dict,
            embedding_dim=6,
            text_embedding_path=path,
            embedding_source="wordnet+llama",
        )
        assert head.embedding_source == "wordnet+llama"
        assert head.text_embedding_path == str(path.resolve())
        assert head.projection.in_features == 6


def test_json_text_embeddings_accept_a_gloss_mapping():
    gloss_dict = {"a": [1, 0], "b": [2, 1]}
    payload = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "semantic.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_text_embeddings(
            path,
            gloss_dict,
            num_classes=3,
            embedding_dim=3,
        )
    assert torch.count_nonzero(loaded[0]) == 0
    assert loaded[1, 0].item() == 1.0
    assert loaded[2, 1].item() == 1.0


def test_invalid_semantic_embedding_rows_are_rejected():
    gloss_dict = {"a": [1, 0]}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.npy"
        np.save(path, np.ones((2, 4), dtype=np.float32))
        try:
            load_text_embeddings(path, gloss_dict, num_classes=2)
        except ValueError as error:
            assert "blank" in str(error)
        else:
            raise AssertionError("non-zero blank embedding was accepted")


def test_semantic_source_requires_precomputed_embeddings():
    gloss_dict = {"a": [1, 0], "b": [2, 1]}
    try:
        TextPrototypeHead(
            num_classes=3,
            feature_dim=8,
            gloss_dict=gloss_dict,
            embedding_source="wordnet+llama",
        )
    except ValueError as error:
        assert "precomputed embeddings" in str(error)
    else:
        raise AssertionError("semantic source silently fell back to lexical")


if __name__ == "__main__":
    test_text_prototype_shapes_and_alignment()
    test_text_alignment_skips_low_confidence_and_sparse_visuals()
    test_precomputed_semantic_embeddings_are_loaded_with_provenance()
    test_json_text_embeddings_accept_a_gloss_mapping()
    test_invalid_semantic_embedding_rows_are_rejected()
    test_semantic_source_requires_precomputed_embeddings()
    print("text prototype tests passed")
