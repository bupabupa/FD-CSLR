from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main


class _Recorder:
    @staticmethod
    def print_log(message):
        return None


def test_best_dev_checkpoint_is_saved_between_snapshot_epochs():
    original_train = main.seq_train
    original_eval = main.seq_eval
    dev_values = iter([40.0, 38.0, 37.0, 37.5])
    saved = []

    def fake_train(*args, **kwargs):
        return None

    def fake_eval(*args, **kwargs):
        return next(dev_values)

    main.seq_train = fake_train
    main.seq_eval = fake_eval
    try:
        processor = object.__new__(main.Processor)
        processor.arg = SimpleNamespace(
            phase="train",
            optimizer_args={"start_epoch": 0},
            num_epoch=4,
            save_interval=5,
            eval_interval=1,
            evaluate_tool="sclite",
            work_dir="unused/",
        )
        processor.recoder = _Recorder()
        processor.data_loader = {"train": None, "dev": None}
        processor.model = None
        processor.optimizer = None
        processor.device = None
        processor.save_model = (
            lambda epoch, path: saved.append((epoch, path))
        )
        processor.start()
    finally:
        main.seq_train = original_train
        main.seq_eval = original_eval

    assert [epoch for epoch, _ in saved] == [0, 1, 2]
    assert saved[-1][1].endswith("dev_37.00_epoch2_model.pt")


def test_best_cal_checkpoint_does_not_touch_selection_split():
    original_train = main.seq_train
    original_eval = main.seq_eval
    cal_values = iter([41.0, 39.0])
    evaluated_modes = []
    saved = []

    def fake_train(*args, **kwargs):
        return None

    def fake_eval(cfg, loader, model, device, mode, *args, **kwargs):
        evaluated_modes.append(mode)
        return next(cal_values)

    main.seq_train = fake_train
    main.seq_eval = fake_eval
    try:
        processor = object.__new__(main.Processor)
        processor.arg = SimpleNamespace(
            phase="train",
            optimizer_args={"start_epoch": 0},
            num_epoch=2,
            save_interval=5,
            eval_interval=1,
            evaluate_tool="sclite",
            checkpoint_split="cal",
            work_dir="unused/",
        )
        processor.recoder = _Recorder()
        processor.data_loader = {"train": None, "cal": None}
        processor.model = None
        processor.optimizer = None
        processor.device = None
        processor.save_model = (
            lambda epoch, path: saved.append((epoch, path))
        )
        processor.start()
    finally:
        main.seq_train = original_train
        main.seq_eval = original_eval

    assert evaluated_modes == ["cal", "cal"]
    assert saved[-1][1].endswith("cal_39.00_epoch1_model.pt")


if __name__ == "__main__":
    test_best_dev_checkpoint_is_saved_between_snapshot_epochs()
    test_best_cal_checkpoint_does_not_touch_selection_split()
    print("best checkpointing tests passed")
