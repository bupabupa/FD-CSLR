import json
import math
import re
from pathlib import Path


DOI_PATTERN = re.compile(r"^10\.[0-9]{4,9}/\S+$")


def _required_text(record, field, context):
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} requires a non-empty {}".format(context, field))
    return value.strip()


def _source_identifier(source, context):
    identifier = source.get("identifier")
    if not isinstance(identifier, dict):
        raise ValueError(
            "{} source requires a DOI or URL identifier".format(context)
        )
    doi = identifier.get("doi")
    url = identifier.get("url")
    if doi is not None:
        doi = str(doi).strip()
        if not DOI_PATTERN.match(doi):
            raise ValueError("{} has an invalid DOI".format(context))
    if url is not None:
        url = str(url).strip()
        if not url.startswith(("https://", "http://")):
            raise ValueError("{} has an invalid source URL".format(context))
    if not doi and not url:
        raise ValueError(
            "{} source requires a DOI or URL identifier".format(context)
        )
    return {"doi": doi or None, "url": url or None}


def _validate_metrics(metrics, context):
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("{} requires non-empty metrics".format(context))
    output = {}
    for name, value in metrics.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "{} metric {} must be numeric".format(context, name)
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                "{} metric {} must be finite".format(context, name)
            )
        output[str(name)] = value
    return output


def validate_external_comparisons(payload, expected_split_sha256):
    """Validate source, protocol and same-backbone comparison constraints."""
    if not isinstance(payload, dict):
        raise ValueError("comparison payload must be a JSON object")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("comparison schema_version must be 1")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("comparison payload requires non-empty entries")

    validated = {}
    for index, entry in enumerate(entries):
        context = "entry[{}]".format(index)
        if not isinstance(entry, dict):
            raise ValueError("{} must be an object".format(context))
        name = _required_text(entry, "name", context)
        if name in validated:
            raise ValueError("comparison entry names must be unique")
        kind = entry.get("kind")
        if kind not in {"ours", "external"}:
            raise ValueError("{} kind must be ours or external".format(context))
        backbone = _required_text(entry, "backbone", context)
        decoder = _required_text(entry, "decoder", context)
        split_sha256 = _required_text(entry, "split_sha256", context)
        comparable = bool(entry.get("comparable", True))
        if comparable and split_sha256 != str(expected_split_sha256):
            raise ValueError(
                "{} is marked comparable but its split hash differs".format(
                    context
                )
            )

        record = {
            "name": name,
            "kind": kind,
            "backbone": backbone,
            "decoder": decoder,
            "split_sha256": split_sha256,
            "comparable": comparable,
            "metrics": _validate_metrics(
                entry.get("metrics"),
                context,
            ),
        }
        if kind == "external":
            source = entry.get("source")
            if not isinstance(source, dict):
                raise ValueError(
                    "{} external result requires source metadata".format(
                        context
                    )
                )
            source_title = _required_text(source, "title", context + " source")
            try:
                year = int(source.get("year"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "{} source year must be an integer".format(context)
                ) from error
            record["source"] = {
                "title": source_title,
                "year": year,
                "identifier": _source_identifier(
                    source,
                    context,
                ),
            }
        else:
            provenance = entry.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(
                    "{} project result requires artifact provenance".format(
                        context
                    )
                )
            record["provenance"] = {
                field: _required_text(provenance, field, context + " provenance")
                for field in (
                    "config_sha256",
                    "checkpoint_sha256",
                    "prediction_sha256",
                )
            }
        validated[name] = record

    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("comparison claims must be a list")
    validated_claims = []
    for index, claim in enumerate(claims):
        context = "claim[{}]".format(index)
        if not isinstance(claim, dict):
            raise ValueError("{} must be an object".format(context))
        candidate = _required_text(claim, "candidate", context)
        competitor = _required_text(claim, "competitor", context)
        metric = _required_text(claim, "metric", context)
        if candidate not in validated or competitor not in validated:
            raise ValueError("{} references an unknown comparison entry".format(
                context
            ))
        candidate_entry = validated[candidate]
        competitor_entry = validated[competitor]
        if not candidate_entry["comparable"] or not competitor_entry["comparable"]:
            raise ValueError("{} uses a non-comparable entry".format(context))
        if candidate_entry["backbone"] != competitor_entry["backbone"]:
            raise ValueError(
                "{} mixes backbones {} and {}; report separate tables".format(
                    context,
                    candidate_entry["backbone"],
                    competitor_entry["backbone"],
                )
            )
        if candidate_entry["decoder"] != competitor_entry["decoder"]:
            raise ValueError(
                "{} mixes decoders {} and {}".format(
                    context,
                    candidate_entry["decoder"],
                    competitor_entry["decoder"],
                )
            )
        if metric not in candidate_entry["metrics"]:
            raise ValueError("{} candidate lacks metric {}".format(context, metric))
        if metric not in competitor_entry["metrics"]:
            raise ValueError(
                "{} competitor lacks metric {}".format(context, metric)
            )
        validated_claims.append({
            "candidate": candidate,
            "competitor": competitor,
            "metric": metric,
            "backbone": candidate_entry["backbone"],
            "decoder": candidate_entry["decoder"],
            "candidate_value": candidate_entry["metrics"][metric],
            "competitor_value": competitor_entry["metrics"][metric],
        })

    comparable_entries = [
        record for record in validated.values() if record["comparable"]
    ]
    groups = {}
    for record in comparable_entries:
        key = "{}|{}".format(record["backbone"], record["decoder"])
        groups.setdefault(key, []).append(record["name"])
    return {
        "schema_version": 1,
        "split_sha256": str(expected_split_sha256),
        "num_entries": len(validated),
        "num_comparable_entries": len(comparable_entries),
        "num_claims": len(validated_claims),
        "same_backbone_decoder_groups": {
            key: sorted(names) for key, names in sorted(groups.items())
        },
        "claims": validated_claims,
        "entries": validated,
    }


def load_and_validate(path, expected_split_sha256):
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    return validate_external_comparisons(
        payload,
        expected_split_sha256=expected_split_sha256,
    )


def write_template(path):
    template = {
        "schema_version": 1,
        "entries": [
            {
                "name": "replace-me",
                "kind": "external",
                "backbone": "resnet18",
                "decoder": "beam:ctcdecode:width10",
                "split_sha256": "REPLACE_WITH_LOCKED_SPLIT_SHA256",
                "comparable": True,
                "metrics": {"wer": 0.0},
                "source": {
                    "title": "REPLACE_WITH_PAPER_TITLE",
                    "year": 2026,
                    "identifier": {
                        "doi": "10.0000/replace-me",
                        "url": None,
                    },
                },
            }
        ],
        "claims": [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
