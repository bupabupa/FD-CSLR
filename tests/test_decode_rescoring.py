import inspect

import torch

from fd_xmp_network import FDXMPModel
from slr_network import SLRModel
from utils.decode import _collapsed_beam, _select_beam_index


class _FakeRescorer:
    def __init__(self, scores):
        self.scores = scores

    def score(self, sequence):
        return self.scores[tuple(sequence)]


def test_rescorer_none_preserves_first_beam():
    scores = torch.tensor([0.1, 0.2])
    assert _select_beam_index(
        [[1], [2]],
        scores,
        rescorer=None,
    ) == 0


def test_ctc_scores_are_negative_log_probabilities():
    rescorer = _FakeRescorer({
        (1,): -10.0,
        (2,): 0.0,
    })
    sequences = [[1], [2]]
    ctc_scores = torch.tensor([0.01, 0.10])
    assert _select_beam_index(
        sequences,
        ctc_scores,
        rescorer=rescorer,
        alpha=0.0,
    ) == 0
    assert _select_beam_index(
        sequences,
        ctc_scores,
        rescorer=rescorer,
        alpha=1.0,
    ) == 1


def test_beam_collapse_removes_blanks_and_repeats():
    beam = torch.tensor([0, 1, 1, 0, 2, 2, 1])
    assert _collapsed_beam(beam, 7, blank_id=0) == [1, 2, 1]


def test_decoder_parameters_default_to_legacy_behavior():
    slr_parameters = inspect.signature(
        SLRModel.__init__
    ).parameters
    fd_xmp_parameters = inspect.signature(
        FDXMPModel.__init__
    ).parameters
    defaults = {
        "beam_width": 10,
        "num_processes": 10,
        "rescorer_path": None,
        "rescorer_alpha": 0.0,
        "rescorer_beta": 0.0,
    }
    for name, expected in defaults.items():
        assert slr_parameters[name].default == expected
        assert fd_xmp_parameters[name].default == expected


if __name__ == "__main__":
    test_rescorer_none_preserves_first_beam()
    test_ctc_scores_are_negative_log_probabilities()
    test_beam_collapse_removes_blanks_and_repeats()
    test_decoder_parameters_default_to_legacy_behavior()
    print("decode rescoring tests passed")
