import torch
import torch.nn.functional as F

from fd_xmp_network import FDXMPModel
from modules.criterions import SeqKD
from slr_network import SLRModel, finite_ctc_mean


class _PerSampleCTCLoss:
    def __init__(self):
        self.calls = 0
        self.log_probs = []

    def __call__(self, log_probs, *args, **kwargs):
        self.calls += 1
        self.log_probs.append(log_probs.detach().clone())
        if self.calls in (2, 4):
            return torch.tensor([float("inf")], device=log_probs.device)
        return log_probs.sum().reshape(1)


class _CTCBranchHarness:
    _ctc_branch_loss = SLRModel._ctc_branch_loss

    def __init__(self):
        self.loss = {"CTCLoss": _PerSampleCTCLoss()}
        self.last_ctc_validity = {}


class _FDXMPCriterionHarness:
    _ctc_branch_loss = SLRModel._ctc_branch_loss
    criterion_calculation = FDXMPModel.criterion_calculation

    def __init__(self):
        self.loss = {
            "CTCLoss": _PerSampleCTCLoss(),
            "distillation": None,
        }
        self.loss_weights = {"SeqCTC": 1.0, "ConvCTC": 1.0}
        self.last_ctc_validity = {}


def test_finite_ctc_mean_masks_only_invalid_samples():
    losses = torch.tensor([1.0, float("inf"), 3.0])
    mean, valid = finite_ctc_mean(losses, "ConvCTC")

    torch.testing.assert_close(mean, torch.tensor(2.0))
    assert valid.tolist() == [True, False, True]


def test_finite_ctc_mean_rejects_an_entirely_invalid_branch():
    losses = torch.tensor([float("inf"), float("nan")])

    try:
        finite_ctc_mean(losses, "ConvCTC")
    except FloatingPointError as error:
        assert "ConvCTC" in str(error)
    else:
        raise AssertionError("non-finite branch should raise")


def test_ctc_invalid_sample_is_excluded_before_backward():
    harness = _CTCBranchHarness()
    logits = torch.zeros(3, 2, 4, requires_grad=True)
    label = torch.tensor([1, 2])
    branch_loss = harness._ctc_branch_loss(
        logits,
        label,
        torch.tensor([3, 2]),
        torch.tensor([1, 1]),
        "SeqCTC",
    )

    branch_loss.backward()

    assert harness.loss["CTCLoss"].calls == 2
    assert harness.last_ctc_validity["SeqCTC"].tolist() == [True, False]
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, 1, :]) == 0


def test_fdxmp_uses_the_same_per_sample_ctc_isolation():
    harness = _FDXMPCriterionHarness()
    sequence_logits = torch.zeros(3, 2, 4, requires_grad=True)
    conv_logits = torch.zeros(3, 2, 4, requires_grad=True)
    ret_dict = {
        "sequence_logits": sequence_logits,
        "conv_logits": conv_logits,
        "feat_len": torch.tensor([3, 2]),
    }
    loss = harness.criterion_calculation(
        ret_dict,
        torch.tensor([1, 2]),
        torch.tensor([1, 1]),
    )
    loss.backward()

    assert harness.loss["CTCLoss"].calls == 4
    assert harness.last_ctc_validity["SeqCTC"].tolist() == [True, False]
    assert harness.last_ctc_validity["ConvCTC"].tolist() == [True, False]
    assert torch.isfinite(sequence_logits.grad).all()
    assert torch.isfinite(conv_logits.grad).all()
    assert torch.count_nonzero(sequence_logits.grad[:, 1, :]) == 0
    assert torch.count_nonzero(conv_logits.grad[:, 1, :]) == 0


def test_fdxmp_criterion_uses_calibrated_log_probs_without_softmax_twice():
    harness = _FDXMPCriterionHarness()
    harness.loss_weights = {"SeqCTC": 1.0}
    sequence_logits = torch.zeros(5, 1, 4, requires_grad=True)
    sequence_log_probs = torch.tensor(
        [
            [[-0.1, -2.3, -3.1, -2.8]],
            [[-0.2, -2.1, -3.0, -2.7]],
            [[-1.0, -0.3, -2.4, -2.9]],
            [[-1.1, -0.4, -2.5, -2.8]],
            [[-0.3, -2.0, -2.9, -2.6]],
        ],
        requires_grad=True,
    )
    ret_dict = {
        "sequence_logits": sequence_logits,
        "sequence_log_probs": sequence_log_probs,
        "conv_logits": sequence_logits,
        "feat_len": torch.tensor([5]),
    }
    loss = harness.criterion_calculation(
        ret_dict,
        torch.tensor([2]),
        torch.tensor([1]),
    )
    loss.backward()

    assert harness.loss["CTCLoss"].calls == 1
    torch.testing.assert_close(
        harness.loss["CTCLoss"].log_probs[0],
        sequence_log_probs.detach(),
    )
    assert sequence_log_probs.grad is not None
    assert sequence_log_probs.grad.abs().sum().item() > 0
    assert torch.count_nonzero(sequence_logits.grad) == 0


def test_seqkd_consumes_normalized_calibrated_log_probs():
    prediction = torch.randn(2, 3, 5)
    reference_logits = torch.randn(2, 3, 5)
    reference_probs = reference_logits.softmax(dim=-1)
    reference_log_probs = reference_probs.log()

    loss = SeqKD(T=1)(
        prediction,
        reference_log_probs,
        ref_is_log_probs=True,
    )
    expected = F.kl_div(
        F.log_softmax(prediction, dim=-1).view(-1, 5),
        reference_probs.view(-1, 5),
        reduction="batchmean",
    )
    torch.testing.assert_close(loss, expected)

    try:
        SeqKD(T=1)(
            prediction,
            reference_log_probs + 1.0,
            ref_is_log_probs=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unnormalized reference log-probs were accepted")


if __name__ == "__main__":
    test_finite_ctc_mean_masks_only_invalid_samples()
    test_finite_ctc_mean_rejects_an_entirely_invalid_branch()
    test_ctc_invalid_sample_is_excluded_before_backward()
    test_fdxmp_uses_the_same_per_sample_ctc_isolation()
    test_fdxmp_criterion_uses_calibrated_log_probs_without_softmax_twice()
    test_seqkd_consumes_normalized_calibrated_log_probs()
    print("CTC loss masking tests passed")
