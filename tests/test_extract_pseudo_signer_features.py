import torch

from scripts.extract_pseudo_signer_features import (
    evenly_spaced_indices,
    pool_frame_embeddings,
    validate_split_access,
)


def test_evenly_spaced_indices_are_deterministic_and_bounded():
    indices = evenly_spaced_indices(100, 16)
    assert len(indices) == 16
    assert len(set(indices)) == 16
    assert indices[0] == 0
    assert indices[-1] == 99
    assert evenly_spaced_indices(3, 16) == [0, 1, 2]


def test_pool_frame_embeddings_averages_and_normalizes():
    features = torch.tensor(
        [
            [
                [3.0, 0.0],
                [0.0, 4.0],
            ],
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
        ]
    )
    pooled = pool_frame_embeddings(features)
    torch.testing.assert_close(
        pooled,
        torch.tensor(
            [
                [0.6, 0.8],
                [1.0, 0.0],
            ]
        ),
    )
    torch.testing.assert_close(
        pooled.norm(dim=-1),
        torch.ones(2),
    )


def test_test_split_requires_explicit_unlock():
    try:
        validate_split_access(["train", "test"])
    except ValueError as error:
        assert "locked" in str(error)
    else:
        raise AssertionError("Test extraction was accepted without unlock")
    assert validate_split_access(
        ["train", "cal", "sel", "test"],
        allow_test=True,
    ) == ["train", "cal", "sel", "test"]


if __name__ == "__main__":
    test_evenly_spaced_indices_are_deterministic_and_bounded()
    test_pool_frame_embeddings_averages_and_normalizes()
    test_test_split_requires_explicit_unlock()
    print("pseudo signer feature extraction tests passed")
