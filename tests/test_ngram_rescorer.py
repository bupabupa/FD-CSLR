import json
import math
import tempfile
from pathlib import Path

import numpy as np

from scripts.build_ngram_rescorer import build_rescorer
from utils.ngram_rescorer import NGramRescorer


def _expect_value_error(callback, message):
    try:
        callback()
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_known_sequence_order_and_finite_empty_score():
    sequences = [[1, 2], [1, 2], [1, 2], [2, 1]]
    scorer = NGramRescorer(
        sequences,
        order=2,
        vocabulary_size=4,
    )
    assert scorer.score([1, 2]) > scorer.score([2, 1])
    assert math.isfinite(scorer.score([]))


def test_rescorer_requires_train_source():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rescorer.json"
        path.write_text(
            json.dumps({
                "source": "D_dev",
                "order": 2,
                "num_classes": 4,
                "sequences": [[1, 2]],
            }),
            encoding="utf-8",
        )
        _expect_value_error(
            lambda: NGramRescorer.from_json(path),
            "source=D_train",
        )


def test_builder_records_strict_train_metadata():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        gloss_path = directory / "gloss_dict.npy"
        np.save(
            gloss_path,
            np.array({
                "BOOK": [1],
                "GO": [2],
            }, dtype=object),
            allow_pickle=True,
        )
        payload = build_rescorer(
            gloss_path,
            {
                "prefix": ["sample"],
                0: {"label": "BOOK GO"},
                1: {"label": "GO"},
            },
            order=3,
            split_sha256="a" * 64,
        )
        assert payload["source"] == "D_train"
        assert payload["split_sha256"] == "a" * 64
        assert payload["num_classes"] == 3
        assert payload["sequences"] == [[1, 2], [2]]
        assert payload["num_sequences"] == 2
        assert payload["num_gloss_tokens"] == 3


if __name__ == "__main__":
    test_known_sequence_order_and_finite_empty_score()
    test_rescorer_requires_train_source()
    test_builder_records_strict_train_metadata()
    print("n-gram rescorer tests passed")
