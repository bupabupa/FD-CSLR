import tempfile
from pathlib import Path

import torch

from scripts.select_effective_checkpoint import select_checkpoint


def _checkpoint(path, value, split_sha256="split"):
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": {"weight": torch.tensor([value])},
            "split_sha256": split_sha256,
        },
        path,
    )


def _dev(path, wer):
    path.write_text(
        "Epoch 39, dev {:.2f}%\n".format(wer),
        encoding="utf-8",
    )


def test_average_is_used_only_for_strict_improvement():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        single = root / "single.pt"
        average = root / "average.pt"
        single_dev = root / "single-dev.txt"
        average_dev = root / "average-dev.txt"
        _checkpoint(single, 1.0)
        _checkpoint(average, 2.0)
        _dev(single_dev, 20.0)
        _dev(average_dev, 19.5)
        improved = select_checkpoint(
            single, single_dev, average, average_dev
        )
        _dev(average_dev, 20.0)
        tied = select_checkpoint(
            single, single_dev, average, average_dev
        )
    assert improved["selected"] == "average"
    assert improved["selected_checkpoint"] == str(average.resolve())
    assert tied["selected"] == "single"
    assert tied["selected_checkpoint"] == str(single.resolve())


def test_split_hash_mismatch_and_missing_hash_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        single = root / "single.pt"
        average = root / "average.pt"
        single_dev = root / "single-dev.txt"
        average_dev = root / "average-dev.txt"
        _checkpoint(single, 1.0, split_sha256="a")
        _checkpoint(average, 2.0, split_sha256="b")
        _dev(single_dev, 20.0)
        _dev(average_dev, 19.0)
        try:
            select_checkpoint(
                single, single_dev, average, average_dev
            )
        except ValueError as error:
            assert "different split hashes" in str(error)
        else:
            raise AssertionError("split hash mismatch was accepted")

        torch.save(
            {
                "model_state_dict": {"weight": torch.zeros(1)},
            },
            average,
        )
        try:
            select_checkpoint(
                single, single_dev, average, average_dev
            )
        except ValueError as error:
            assert "missing split_sha256" in str(error)
        else:
            raise AssertionError("missing split hash was accepted")


if __name__ == "__main__":
    test_average_is_used_only_for_strict_improvement()
    test_split_hash_mismatch_and_missing_hash_are_rejected()
    print("effective checkpoint selection tests passed")
