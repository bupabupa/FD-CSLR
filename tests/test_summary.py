import json
import tempfile
from pathlib import Path

from scripts.summarize_predictions import (
    bootstrap_difference,
    bootstrap_wer,
    grouped_statistics,
    holm_adjust,
    read_predictions,
    sentence_statistics,
)


def test_bootstrap_summary():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "predictions-test.jsonl"
        rows = [
            {
                "sample_id": "a",
                "reference_ids": [1, 2],
                "hypothesis_ids": [1, 2],
                "signer": "Signer04",
            },
            {
                "sample_id": "b",
                "reference_ids": [1, 2],
                "hypothesis_ids": [1],
                "signer": "Signer07",
            },
            {
                "sample_id": "c",
                "reference_ids": [1, 2],
                "hypothesis_ids": [1, 3],
                "signer": "Signer08",
            },
        ]
        with open(path, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
        samples = read_predictions(path)
    ids, edits, lengths = sentence_statistics(samples, num_classes=4)
    assert ids == ["a", "b", "c"]
    assert edits.tolist() == [0, 1, 1]
    assert lengths.tolist() == [2, 2, 2]
    groups = grouped_statistics(samples, num_classes=4, field="signer")
    assert groups["Signer04"]["wer"] == 0
    assert groups["Signer07"]["wer"] == 50
    assert groups["Signer08"]["wer"] == 50
    interval = bootstrap_wer(edits, lengths, 100, 0)
    assert interval["lower_95"] <= interval["upper_95"]
    difference = bootstrap_difference(
        edits,
        lengths,
        edits,
        lengths,
        100,
        0,
    )
    assert difference["mean"] == 0
    assert difference["p_value_one_sided"] == 1
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


if __name__ == "__main__":
    test_bootstrap_summary()
    print("prediction summary tests passed")
