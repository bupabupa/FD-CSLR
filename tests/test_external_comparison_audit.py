import copy

from evaluation.external_comparison_audit import (
    validate_external_comparisons,
)


SPLIT = "a" * 64


def _payload():
    return {
        "schema_version": 1,
        "entries": [
            {
                "name": "ours-r18",
                "kind": "ours",
                "backbone": "resnet18",
                "decoder": "beam:ctcdecode:width10",
                "split_sha256": SPLIT,
                "comparable": True,
                "metrics": {"wer": 20.0, "uwer": 0.4},
                "provenance": {
                    "config_sha256": "b" * 64,
                    "checkpoint_sha256": "c" * 64,
                    "prediction_sha256": "d" * 64,
                },
            },
            {
                "name": "cap-slr-r18",
                "kind": "external",
                "backbone": "resnet18",
                "decoder": "beam:ctcdecode:width10",
                "split_sha256": SPLIT,
                "comparable": True,
                "metrics": {"wer": 21.0, "uwer": 0.5},
                "source": {
                    "title": "External result",
                    "year": 2025,
                    "identifier": {
                        "doi": "10.1000/example.2025.1",
                        "url": None,
                    },
                },
            },
        ],
        "claims": [
            {
                "candidate": "ours-r18",
                "competitor": "cap-slr-r18",
                "metric": "wer",
            }
        ],
    }


def test_same_backbone_same_decoder_claim_is_accepted():
    report = validate_external_comparisons(_payload(), SPLIT)
    assert report["num_claims"] == 1
    assert report["claims"][0]["backbone"] == "resnet18"


def test_cross_backbone_claim_is_rejected():
    payload = _payload()
    payload["entries"][1]["backbone"] = "resnet34"
    try:
        validate_external_comparisons(payload, SPLIT)
    except ValueError as error:
        assert "mixes backbones" in str(error)
    else:
        raise AssertionError("cross-backbone SOTA claim was accepted")


def test_external_source_requires_doi_or_url():
    payload = _payload()
    payload["entries"][1]["source"]["identifier"] = {
        "doi": None,
        "url": None,
    }
    try:
        validate_external_comparisons(payload, SPLIT)
    except ValueError as error:
        assert "DOI or URL" in str(error)
    else:
        raise AssertionError("source without DOI or URL was accepted")


def test_non_comparable_entry_cannot_enter_a_claim():
    payload = copy.deepcopy(_payload())
    payload["entries"][1]["split_sha256"] = "e" * 64
    payload["entries"][1]["comparable"] = False
    try:
        validate_external_comparisons(payload, SPLIT)
    except ValueError as error:
        assert "non-comparable" in str(error)
    else:
        raise AssertionError("non-comparable entry entered a claim")


if __name__ == "__main__":
    test_same_backbone_same_decoder_claim_is_accepted()
    test_cross_backbone_claim_is_rejected()
    test_external_source_requires_doi_or_url()
    test_non_comparable_entry_cannot_enter_a_claim()
    print("external comparison audit tests passed")
