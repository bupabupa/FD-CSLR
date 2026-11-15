import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from fd_xmp_network import FDXMPModel, signer_multi_positive_loss


def test_cross_signer_info_nce_and_gradients():
    anchors = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )
    classes = torch.tensor([3, 4])
    signers = torch.tensor([1, 1])
    memory = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]],
        dtype=torch.float32,
    )
    memory_classes = torch.tensor([3, 4, 5])
    memory_signers = torch.tensor([2, 2, 3])
    memory_valid = torch.tensor([True, True, True])

    loss, coverage = signer_multi_positive_loss(
        anchors,
        classes,
        signers,
        memory,
        memory_classes,
        memory_signers,
        memory_valid,
        temperature=0.1,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    torch.testing.assert_close(coverage, torch.tensor(1.0))
    loss.backward()
    assert anchors.grad is not None
    assert anchors.grad.abs().sum().item() > 0.0
    print("cross-signer InfoNCE test passed", flush=True)


def test_missing_cross_signer_positive_is_skipped():
    anchors = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss, coverage = signer_multi_positive_loss(
        anchors,
        torch.tensor([3]),
        torch.tensor([1]),
        torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        torch.tensor([3, 4]),
        torch.tensor([1, 2]),
        torch.tensor([True, True]),
        temperature=0.1,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(coverage, torch.tensor(0.0))
    print("missing-positive skip test passed", flush=True)


def test_occurrence_collection_and_memory_update():
    dummy = SimpleNamespace(
        signer_inv_confidence=0.1,
        signer_inv_min_mass=0.05,
    )
    features = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0, 0.0]],
            [[9.0, 0.0, 0.0, 0.0]],
        ]
    )
    gamma = torch.zeros(3, 1, 4)
    gamma[0, 0, 2] = 1.0
    gamma[1, 0, 2] = 3.0
    gamma[2, 0, 2] = 2.0
    confidence = torch.ones(3, 1)
    occurrences = FDXMPModel._collect_signer_occurrences(
        dummy,
        features,
        gamma,
        confidence,
        torch.tensor([1]),
        torch.tensor([2]),
        torch.tensor([1]),
        torch.tensor([3]),
    )
    embedding, classes, signers = occurrences
    torch.testing.assert_close(
        embedding, torch.tensor([[28.0 / 6.0, 0.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(classes, torch.tensor([2]))
    torch.testing.assert_close(signers, torch.tensor([1]))

    memory = SimpleNamespace(
        signer_occurrence_embeddings=torch.zeros(2, 4),
        signer_occurrence_classes=torch.full((2,), -1, dtype=torch.long),
        signer_occurrence_signers=torch.full((2,), -1, dtype=torch.long),
        signer_occurrence_valid=torch.zeros(2, dtype=torch.bool),
        signer_occurrence_ptr=torch.zeros((), dtype=torch.long),
    )
    FDXMPModel._update_signer_occurrence_memory(
        memory, embedding.detach(), classes, signers
    )
    assert memory.signer_occurrence_valid.tolist() == [True, False]
    torch.testing.assert_close(memory.signer_occurrence_classes, torch.tensor([2, -1]))
    assert memory.signer_occurrence_ptr.item() == 1
    print("signer occurrence memory test passed", flush=True)


def test_new_losses_have_gradients():
    dummy = SimpleNamespace(
        loss_weights={"SignerInv": 0.05, "SignerOrth": 0.1}
    )
    inv = torch.tensor(2.0, requires_grad=True)
    orth = torch.tensor(0.5, requires_grad=True)
    loss = FDXMPModel.criterion_calculation(
        dummy,
        {
            "sequence_logits": torch.zeros(1),
            "signer_inv_loss": inv,
            "signer_orthogonal_loss": orth,
        },
        None,
        None,
    )
    loss.backward()
    assert inv.grad is not None
    assert orth.grad is not None
    torch.testing.assert_close(inv.grad, torch.tensor(0.05))
    torch.testing.assert_close(orth.grad, torch.tensor(0.1))
    print("signer loss wiring test passed", flush=True)


def test_initialized_branches_reconstruct_temporal_features():
    model = FDXMPModel(
        num_classes=3,
        c2d_type="resnet18",
        conv_type=2,
        use_bn=0,
        hidden_size=8,
        gloss_dict={"a": [1, "a"]},
        loss_weights={"SeqCTC": 1.0, "ConvCTC": 1.0},
        weight_norm=False,
        share_classifier=False,
        signer_film=True,
        max_signer_id=3,
        signer_rank=2,
    ).eval()
    assert not any(
        "signer_occurrence" in key for key in model.state_dict()
    )
    video = torch.randn(1, 32, 3, 224, 224)
    with torch.no_grad():
        output = model(
            video,
            torch.tensor([32]),
            decode=False,
            signer_ids=torch.tensor([1]),
        )
    reconstructed = (
        output["signer_branch_features"]
        + output["conditioned_gloss_features"]
    )
    torch.testing.assert_close(
        reconstructed, output["temporal_features"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        output["signer_gamma"], torch.zeros_like(output["signer_gamma"])
    )
    torch.testing.assert_close(
        output["signer_beta"], torch.zeros_like(output["signer_beta"])
    )
    print("initialized signer branches preserve identity", flush=True)


if __name__ == "__main__":
    test_cross_signer_info_nce_and_gradients()
    test_missing_cross_signer_positive_is_skipped()
    test_occurrence_collection_and_memory_update()
    test_new_losses_have_gradients()
    test_initialized_branches_reconstruct_temporal_features()
    print("signer invariance loss tests passed", flush=True)
