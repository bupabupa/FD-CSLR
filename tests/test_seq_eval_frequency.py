import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dataset.splits import load_split_info
from evaluation.frequency_metrics import gloss_occurrence_counts
from seq_scripts import seq_eval


ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = ROOT / "preprocess" / "phoenix2014" / "gloss_dict.npy"
SPLIT_MANIFEST = ROOT / "preprocess" / "phoenix2014" / "split_manifest.json"


class DummyDataset:
    def __init__(self, gloss_dict):
        self.dict = gloss_dict


class DummyLoader:
    def __init__(self, gloss_dict):
        self.dataset = DummyDataset(gloss_dict)
        self.batch = None

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1


class DummyDevice:
    @staticmethod
    def data_to_device(value):
        return value


class DummyModel:
    def __init__(self, recognized, conv):
        self.recognized = recognized
        self.conv = conv

    def eval(self):
        return self

    def __call__(self, video, video_length, label=None, label_lgt=None):
        return {
            "recognized_sents": self.recognized,
            "conv_sents": self.conv,
        }


class DummyRecorder:
    @staticmethod
    def record_timer(name):
        return None

    @staticmethod
    def print_log(message, path=None):
        return None


def test_seq_eval_writes_frequency_report():
    gloss_dict = np.load(DICT_PATH, allow_pickle=True).item()
    glosses = list(gloss_dict)[:2]
    label_ids = [int(gloss_dict[gloss][0]) for gloss in glosses]
    loader = DummyLoader(gloss_dict)
    loader.batch = (
        torch.zeros(1, 3),
        torch.tensor([3]),
        torch.tensor(label_ids),
        torch.tensor([2]),
        ("sample-1|signer",),
    )
    decoded = [(gloss, index) for index, gloss in enumerate(glosses)]
    model = DummyModel([decoded], [decoded])
    cfg = SimpleNamespace(
        split_manifest=str(SPLIT_MANIFEST),
        dataset_info={
            "dict_path": str(DICT_PATH),
            "evaluation_dir": "./evaluation/slr_eval",
            "evaluation_prefix": "phoenix2014-groundtruth",
        }
    )

    with tempfile.TemporaryDirectory() as directory:
        work_dir = directory + "/"
        with patch("seq_scripts.evaluate", return_value=42.0) as evaluate:
            score = seq_eval(
                cfg,
                loader,
                model,
                DummyDevice(),
                mode="dev",
                epoch=0,
                work_dir=work_dir,
                recoder=DummyRecorder(),
                evaluate_tool="python",
            )
        assert score == 42.0
        assert evaluate.call_count == 2
        with open(
            Path(directory) / "frequency-metrics-dev.json",
            encoding="utf-8",
        ) as report_file:
            report = json.load(report_file)
        assert report["overall"]["true_positive"] == 2
        assert report["overall"]["support"] == 2
        assert report["decode"]["unknown_hypothesis_tokens"] == 0
        split_train = load_split_info(
            SPLIT_MANIFEST.parent, "train", SPLIT_MANIFEST
        )
        expected_train_counts = gloss_occurrence_counts(
            split_train, gloss_dict
        )[1:]
        assert report["per_class"]["train_counts"] == (
            expected_train_counts.tolist()
        )
        with open(
            Path(directory) / "predictions-dev.jsonl",
            encoding="utf-8",
        ) as predictions:
            prediction = json.loads(predictions.readline())
        assert prediction["sample_id"] == "sample-1"
        assert prediction["reference_ids"] == label_ids
        assert prediction["hypothesis_ids"] == label_ids
        assert prediction["hypothesis_glosses"] == glosses


if __name__ == "__main__":
    test_seq_eval_writes_frequency_report()
    print("seq_eval frequency integration test passed")
