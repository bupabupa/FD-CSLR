import torch

from seq_scripts import seq_train


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


class _DummyRecorder:
    log_interval = 50

    def print_log(self, *args, **kwargs):
        pass


class _NonFiniteModel:
    supports_signer = False

    def train(self):
        return self

    def __call__(self, *args, **kwargs):
        return {}

    def criterion_calculation(self, *args, **kwargs):
        return torch.tensor(float("inf"), requires_grad=True)


def test_non_finite_training_loss_fails_fast():
    data = (
        torch.zeros(1),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([0]),
        ("sample|sample|Signer01|sample",),
    )
    loader = _DummyLoader([data])
    try:
        seq_train(
            loader,
            _NonFiniteModel(),
            _DummyOptimizer(),
            _DummyDevice(),
            epoch_idx=3,
            recoder=_DummyRecorder(),
        )
    except FloatingPointError as error:
        assert "epoch 3" in str(error)
        assert "batch 0" in str(error)
    else:
        raise AssertionError("non-finite loss did not fail fast")


if __name__ == "__main__":
    test_non_finite_training_loss_fails_fast()
    print("training finite-loss test passed")
