import torch

from scripts.profile_model_cost import (
    parameter_counts,
    synthetic_gloss_dict,
    temporal_frame_floor,
    validate_profile_request,
)


def test_parameter_counts_separate_frozen_parameters():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.Linear(3, 2),
    )
    model[1].weight.requires_grad_(False)
    counts = parameter_counts(model)
    assert counts["parameters"] == 23
    assert counts["trainable_parameters"] == 17
    assert counts["frozen_parameters"] == 6
    assert 0.0 < counts["trainable_ratio"] < 1.0


def test_synthetic_gloss_dictionary_covers_nonblank_classes():
    gloss_dict = synthetic_gloss_dict(5)
    assert sorted(value[0] for value in gloss_dict.values()) == [1, 2, 3, 4]
    assert len(set(gloss_dict)) == 4


def test_profile_request_rejects_shapes_the_temporal_encoder_cannot_use():
    assert [
        temporal_frame_floor(value)
        for value in (0, 1, 2)
    ] == [3, 6, 16]
    validated = validate_profile_request(2, 1, 32, 224, 224, 1, 2)
    assert validated["frames"] == 32
    try:
        validate_profile_request(2, 1, 8, 224, 224, 1, 2)
    except ValueError as error:
        assert "at least 16" in str(error)
    else:
        raise AssertionError("invalid temporal shape was accepted")


if __name__ == "__main__":
    test_parameter_counts_separate_frozen_parameters()
    test_synthetic_gloss_dictionary_covers_nonblank_classes()
    test_profile_request_rejects_shapes_the_temporal_encoder_cannot_use()
    print("model cost profile tests passed")
