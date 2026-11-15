import torch
import torch.nn as nn
import torch.nn.functional as F

from fd_xmp_network import FDXMPModel


class _SignerProbeHarness:
    _signer_probe_logits = FDXMPModel._signer_probe_logits


def test_signer_ce_and_gloss_adversary_use_separate_gradient_paths():
    torch.manual_seed(23)
    strength = 0.2
    harness = _SignerProbeHarness()
    harness.signer_classifier = nn.Linear(4, 3, bias=False)
    harness.gloss_signer_classifier = nn.Linear(4, 3, bias=False)
    harness.gloss_signer_classifier.load_state_dict(
        harness.signer_classifier.state_dict()
    )
    harness.signer_adversarial_strength = strength

    signer_summary = torch.randn(2, 4, requires_grad=True)
    gloss_summary = torch.randn(2, 4, requires_grad=True)
    targets = torch.tensor([0, 2])

    signer_logits, gloss_signer_logits = harness._signer_probe_logits(
        signer_summary, gloss_summary
    )
    signer_loss = F.cross_entropy(signer_logits, targets)
    gloss_signer_loss = F.cross_entropy(gloss_signer_logits, targets)

    signer_loss.backward(retain_graph=True)
    ordinary_signer_grad = signer_summary.grad.detach().clone()
    assert gloss_summary.grad is None

    gloss_signer_loss.backward()
    adversarial_gloss_grad = gloss_summary.grad.detach().clone()

    reference_summary = signer_summary.detach().clone().requires_grad_(True)
    reference_classifier = nn.Linear(4, 3, bias=False)
    reference_classifier.load_state_dict(
        harness.signer_classifier.state_dict()
    )
    reference_logits = reference_classifier(reference_summary)
    F.cross_entropy(reference_logits, targets).backward()

    reference_gloss_summary = (
        gloss_summary.detach().clone().requires_grad_(True)
    )
    gloss_reference_classifier = nn.Linear(4, 3, bias=False)
    gloss_reference_classifier.load_state_dict(
        harness.gloss_signer_classifier.state_dict()
    )
    gloss_reference_logits = gloss_reference_classifier(
        reference_gloss_summary
    )
    F.cross_entropy(gloss_reference_logits, targets).backward()

    torch.testing.assert_close(
        ordinary_signer_grad,
        reference_summary.grad,
    )
    torch.testing.assert_close(
        adversarial_gloss_grad,
        -strength * reference_gloss_summary.grad,
    )

if __name__ == "__main__":
    test_signer_ce_and_gloss_adversary_use_separate_gradient_paths()
    print("signer branch semantics tests passed")
