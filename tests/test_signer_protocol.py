import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from dataset.dataloader_video import BaseFeeder
from dataset.signer_protocol import (
    adjusted_rand_index,
    assign_pseudo_signers,
    cluster_pseudo_signers,
    load_signer_map,
)


def test_ari_is_one_for_identical_partitions():
    first = [0, 0, 1, 1, 2]
    assert adjusted_rand_index(first, first) == 1.0
    assert adjusted_rand_index(first, [2, 2, 1, 1, 0]) == 1.0


def test_pseudo_signer_clustering_is_reproducible_and_covers_samples():
    fileids = ["a", "b", "c", "d", "e", "f"]
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [-10.0, 10.0],
            [-10.1, 10.0],
        ],
        dtype=np.float32,
    )
    first = cluster_pseudo_signers(
        fileids,
        embeddings,
        clusters=3,
        seeds=(0, 1, 2),
    )
    second = cluster_pseudo_signers(
        fileids,
        embeddings,
        clusters=3,
        seeds=(0, 1, 2),
    )
    assert first["mapping"] == second["mapping"]
    assert set(first["mapping"]) == set(fileids)
    assert len(set(first["mapping"].values())) == 3
    assert first["report"]["mean_pairwise_ari"] == 1.0


def test_signer_override_is_parsed_by_collate():
    batch = [
        (
            torch.zeros(8, 3, 4, 4),
            torch.tensor([1]),
            "a|a|Signer04|a|__signer_id=12",
        ),
        (
            torch.zeros(6, 3, 4, 4),
            torch.tensor([2]),
            "b|b|Signer07|b|__signer_id=3",
        ),
    ]
    collated = BaseFeeder.collate_fn(batch)
    assert collated[4].tolist() == [12, 3]


def test_signer_map_requires_multiple_clusters():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "map.json"
        path.write_text(json.dumps({"a": 0, "b": 0}), encoding="utf-8")
        try:
            load_signer_map(path)
        except ValueError as error:
            assert "two clusters" in str(error)
        else:
            raise AssertionError("single-cluster signer map was accepted")


def test_unseen_records_are_assigned_to_training_centers():
    fit_fileids = ["a", "b", "c", "d"]
    fit_embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ],
        dtype=np.float32,
    )
    fitted = cluster_pseudo_signers(
        fit_fileids,
        fit_embeddings,
        clusters=2,
        seeds=(0, 1),
    )
    assigned = assign_pseudo_signers(
        ["cal-a", "sel-a", "test-a"],
        np.asarray(
            [
                [0.05, 0.0],
                [10.05, 10.0],
                [9.95, 10.0],
            ],
            dtype=np.float32,
        ),
        fitted["centers"],
    )
    values = assigned["mapping"]
    assert values["cal-a"] != values["sel-a"]
    assert values["sel-a"] == values["test-a"]
    assert assigned["report"]["num_samples"] == 3


if __name__ == "__main__":
    test_ari_is_one_for_identical_partitions()
    test_pseudo_signer_clustering_is_reproducible_and_covers_samples()
    test_signer_override_is_parsed_by_collate()
    test_signer_map_requires_multiple_clusters()
    test_unseen_records_are_assigned_to_training_centers()
    print("signer protocol tests passed")
