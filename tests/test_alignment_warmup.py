import torch
from unittest.mock import patch

from fd_xmp_network import FDXMPModel
from seq_scripts import (
    _gradient_norm_without_clipping,
    _nonfinite_parameter_names,
    seq_train,
)


class _WarmupHarness:
    set_training_epoch = FDXMPModel.set_training_epoch
    _alignment_warmup_factor = FDXMPModel._alignment_warmup_factor
    criterion_calculation = FDXMPModel.criterion_calculation

    def __init__(self):
        self.proto_align_warmup_epochs = 5
        self.text_align_warmup_epochs = 10
        self._training_epoch = None
        self.loss_weights = {"ProtoAlign": 0.01, "TextAlign": 0.001}
        self.loss = {}


class _DummyLoader(list):
    @property
    def dataset(self):
        return self


class _DummyDevice:
    @staticmethod
    def data_to_device(value):
        return value


class _DummyOptimizer:
    def __init__(self):
        self.optimizer = type(
            "InnerOptimizer",
            (),
            {
                "param_groups": [{"lr": 1e-4}],
                "zero_grad": lambda self: None,
                "step": lambda self: None,
            },
        )()
        self.scheduler = type(
            "Scheduler",
            (),
            {"step": lambda self: None},
        )()

    def zero_grad(self):
        pass

    def step(self):
        pass


class _CountingOptimizer(_DummyOptimizer):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.zero_grad_count = 0

    def zero_grad(self):
        self.zero_grad_count += 1

    def step(self):
        self.step_count += 1


class _DummyRecorder:
    log_interval = 50

    def print_log(self, *args, **kwargs):
        pass


class _EpochRecordingModel:
    supports_signer = False

    def __init__(self):
        self.epochs = []

    def train(self):
        return self

    def set_training_epoch(self, epoch):
        self.epochs.append(epoch)

    def __call__(self, *args, **kwargs):
        return {}

    def criterion_calculation(self, *args, **kwargs):
        return torch.tensor(0.0, requires_grad=True)


class _ClippingModel(torch.nn.Module):
    supports_signer = False

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def set_training_epoch(self, epoch):
        pass

    def forward(self, *args, **kwargs):
        return {}

    def criterion_calculation(self, *args, **kwargs):
        return self.weight * 2.0


def test_alignment_warmup_schedule():
    model = _WarmupHarness()
    assert model._alignment_warmup_factor("ProtoAlign") == 1.0
    assert model._alignment_warmup_factor("TextAlign") == 1.0

    model.set_training_epoch(0)
    assert model._alignment_warmup_factor("ProtoAlign") == 0.0
    assert model._alignment_warmup_factor("TextAlign") == 0.0

    model.set_training_epoch(2)
    assert model._alignment_warmup_factor("ProtoAlign") == 2 / 5
    assert model._alignment_warmup_factor("TextAlign") == 2 / 10

    model.set_training_epoch(5)
    assert model._alignment_warmup_factor("ProtoAlign") == 1.0
    assert model._alignment_warmup_factor("TextAlign") == 0.5

    model.set_training_epoch(20)
    assert model._alignment_warmup_factor("ProtoAlign") == 1.0
    assert model._alignment_warmup_factor("TextAlign") == 1.0


def test_criterion_scales_alignment_losses_only():
    model = _WarmupHarness()
    output = {
        "sequence_logits": torch.zeros(1, requires_grad=True),
        "proto_align_loss": torch.tensor(2.0),
        "text_align_loss": torch.tensor(3.0),
    }

    model.set_training_epoch(0)
    torch.testing.assert_close(
        model.criterion_calculation(output, None, None),
        torch.tensor(0.0),
    )

    model.set_training_epoch(2)
    expected = 0.01 * (2 / 5) * 2.0 + 0.001 * (2 / 10) * 3.0
    torch.testing.assert_close(
        model.criterion_calculation(output, None, None),
        torch.tensor(expected),
    )


def test_zero_warmup_preserves_fixed_weights():
    model = _WarmupHarness()
    model.proto_align_warmup_epochs = 0
    model.text_align_warmup_epochs = 0
    model.set_training_epoch(0)
    output = {
        "sequence_logits": torch.zeros(1, requires_grad=True),
        "proto_align_loss": torch.tensor(2.0),
        "text_align_loss": torch.tensor(3.0),
    }

    torch.testing.assert_close(
        model.criterion_calculation(output, None, None),
        torch.tensor(0.01 * 2.0 + 0.001 * 3.0),
    )


def test_seq_train_sets_epoch_before_batch():
    model = _EpochRecordingModel()
    data = (
        torch.zeros(1),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([0]),
        ("sample|sample|Signer01|sample",),
    )

    seq_train(
        _DummyLoader([data]),
        model,
        _DummyOptimizer(),
        _DummyDevice(),
        epoch_idx=7,
        recoder=_DummyRecorder(),
    )
    assert model.epochs == [7]


def test_gradient_accumulation_steps_optimizer_once_per_group():
    model = _EpochRecordingModel()
    data = (
        torch.zeros(1),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([0]),
        ("sample|sample|Signer01|sample",),
    )
    optimizer = _CountingOptimizer()

    seq_train(
        _DummyLoader([data, data, data]),
        model,
        optimizer,
        _DummyDevice(),
        epoch_idx=0,
        recoder=_DummyRecorder(),
        grad_accum_steps=2,
    )
    assert optimizer.step_count == 2
    assert optimizer.zero_grad_count == 3


def test_gradient_clipping_applied_before_optimizer_step():
    model = _ClippingModel()
    data = (
        torch.zeros(1),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([0]),
        ("sample|sample|Signer01|sample",),
    )
    optimizer = _CountingOptimizer()

    with patch(
        "seq_scripts.torch.nn.utils.clip_grad_norm_",
        return_value=torch.tensor(2.0),
    ) as clip:
        seq_train(
            _DummyLoader([data]),
            model,
            optimizer,
            _DummyDevice(),
            epoch_idx=0,
            recoder=_DummyRecorder(),
            grad_clip_norm=5.0,
        )

    clip.assert_called_once()
    assert clip.call_args[0][1] == 5.0
    assert optimizer.step_count == 1


def test_nonfinite_preclip_gradient_blocks_optimizer_step():
    model = _ClippingModel()
    model.weight.grad = torch.tensor(float("nan"))
    data = (
        torch.zeros(1),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([0]),
        ("sample|sample|Signer01|sample",),
    )
    optimizer = _CountingOptimizer()

    norm = _gradient_norm_without_clipping(model.parameters())
    assert not torch.isfinite(norm)
    assert _nonfinite_parameter_names(model, gradient=True) == ["weight"]

    with patch(
        "seq_scripts._save_nonfinite_batch",
        return_value="/tmp/nonfinite.pt",
    ):
        try:
            seq_train(
                _DummyLoader([data]),
                model,
                optimizer,
                _DummyDevice(),
                epoch_idx=0,
                recoder=_DummyRecorder(),
                grad_clip_norm=5.0,
            )
        except FloatingPointError as error:
            assert "non-finite gradient parameters=['weight']" in str(error)
        else:
            raise AssertionError("non-finite gradients should raise")

    assert optimizer.step_count == 0


if __name__ == "__main__":
    test_alignment_warmup_schedule()
    test_criterion_scales_alignment_losses_only()
    test_zero_warmup_preserves_fixed_weights()
    test_seq_train_sets_epoch_before_batch()
    test_gradient_accumulation_steps_optimizer_once_per_group()
    test_gradient_clipping_applied_before_optimizer_step()
    test_nonfinite_preclip_gradient_blocks_optimizer_step()
    print("alignment warmup tests passed")
