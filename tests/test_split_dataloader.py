from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import main
from dataset.dataloader_video import BaseFeeder


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "preprocess" / "phoenix2014"
MANIFEST = PREPROCESS / "split_manifest.json"
LEAVE_SIGNER01_MANIFEST = (
    PREPROCESS
    / "leave_one_signers"
    / "Signer01"
    / "split_manifest.json"
)


def test_split_dataloader_uses_manifest():
    gloss_dict = np.load(
        PREPROCESS / "gloss_dict.npy", allow_pickle=True
    ).item()
    dataset_root = "/vd_datasets/phoenix2014-release/phoenix-2014-multisigner"
    common = {
        "prefix": dataset_root,
        "gloss_dict": gloss_dict,
        "mode": "train",
        "transform_mode": False,
        "datatype": "video",
    }
    full_train = BaseFeeder(**common)
    filtered_train = BaseFeeder(
        **common,
        split_manifest=str(MANIFEST),
        split_name="train",
    )
    calibration = BaseFeeder(
        **common,
        split_manifest=str(MANIFEST),
        split_name="cal",
    )
    assert len(full_train) == 5671
    assert len(filtered_train) == 5104
    assert len(calibration) == 567
    assert filtered_train.inputs_list[0]["folder"].startswith("train/")
    assert calibration.inputs_list[0]["folder"].startswith("train/")
    batch = [
        (
            torch.zeros(8, 3, 4, 4),
            torch.tensor([1]),
            "a|a|Signer04|a",
        ),
        (
            torch.zeros(6, 3, 4, 4),
            torch.tensor([2]),
            "b|b|Signer07|b",
        ),
    ]
    collated = BaseFeeder.collate_fn(batch)
    assert collated[4].tolist() == [4, 7]
    assert collated[-1][0].endswith("Signer04|a")


def test_processor_can_restrict_dataset_views():
    constructed = []

    class DummyFeeder:
        collate_fn = None

        def __init__(self, **kwargs):
            constructed.append(kwargs)

    processor = object.__new__(main.Processor)
    processor.arg = SimpleNamespace(
        feeder="unused",
        feeder_args={"mode": "train", "datatype": "video"},
        dataset_info={"dataset_root": "/tmp/phoenix"},
        split_manifest=str(MANIFEST),
        load_splits=["cal", "dev"],
    )
    processor.dataset = {}
    processor.data_loader = {}
    processor.gloss_dict = {}
    processor.build_dataloader = (
        lambda dataset, mode, train_flag: {"mode": mode}
    )

    with patch.object(main, "import_class", return_value=DummyFeeder):
        processor.load_data()

    assert set(processor.data_loader) == {"cal", "dev"}
    assert "test" not in processor.data_loader
    assert {
        (item["split_name"], item["mode"]) for item in constructed
    } == {("cal", "train"), ("sel", "dev")}


def test_leave_one_signer_dataloader_isolates_signers():
    gloss_dict = np.load(
        PREPROCESS / "gloss_dict.npy", allow_pickle=True
    ).item()
    common = {
        "prefix": "/vd_datasets/phoenix2014-release/phoenix-2014-multisigner",
        "gloss_dict": gloss_dict,
        "mode": "train",
        "transform_mode": False,
        "datatype": "video",
        "split_manifest": str(LEAVE_SIGNER01_MANIFEST),
    }
    training = BaseFeeder(**common, split_name="train")
    calibration = BaseFeeder(**common, split_name="cal")
    selection = BaseFeeder(**common, split_name="sel")
    test = BaseFeeder(**common, split_name="test")
    assert (len(training), len(calibration), len(selection), len(test)) == (
        3776,
        420,
        590,
        885,
    )
    for view in (training, calibration, selection, test):
        assert all(
            item["folder"].startswith("train/")
            for item in view.inputs_list.values()
            if isinstance(item, dict)
        )
    for view in (selection, test):
        assert all(
            item["signer"] == "Signer01"
            for item in view.inputs_list.values()
            if isinstance(item, dict)
        )
    for view in (training, calibration):
        assert all(
            item["signer"] != "Signer01"
            for item in view.inputs_list.values()
            if isinstance(item, dict)
        )


def test_processor_default_views_follow_phase():
    constructed = []

    class DummyFeeder:
        collate_fn = None

        def __init__(self, **kwargs):
            constructed.append(kwargs)

    def load(phase, eval_splits=("dev", "test")):
        processor = object.__new__(main.Processor)
        processor.arg = SimpleNamespace(
            feeder="unused",
            feeder_args={"mode": "train", "datatype": "video"},
            dataset_info={"dataset_root": "/tmp/phoenix"},
            split_manifest=str(MANIFEST),
            phase=phase,
            eval_splits=list(eval_splits),
            load_splits=None,
        )
        processor.dataset = {}
        processor.data_loader = {}
        processor.gloss_dict = {}
        processor.build_dataloader = (
            lambda dataset, mode, train_flag: {"mode": mode}
        )
        with patch.object(main, "import_class", return_value=DummyFeeder):
            processor.load_data()
        return set(processor.data_loader)

    assert load("train") == {"train", "train_eval", "dev"}
    assert load("test") == {"dev", "test"}
    assert load("features") == {"train_eval", "dev", "test"}


if __name__ == "__main__":
    test_split_dataloader_uses_manifest()
    test_processor_can_restrict_dataset_views()
    test_leave_one_signer_dataloader_isolates_signers()
    test_processor_default_views_follow_phase()
    print("split dataloader test passed")
