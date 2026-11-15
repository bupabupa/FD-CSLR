import argparse
import json
from pathlib import Path

from sequence_normalization import aggregate_wer, normalize_token_sequences


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESS = ROOT / "evaluation" / "slr_eval" / "preprocess.sh"


def load_predictions(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument(
        "--preprocess-script",
        default=str(DEFAULT_PREPROCESS),
    )
    args = parser.parse_args()
    results = {}
    for path in args.predictions:
        records = load_predictions(path)
        references = [
            record["reference_glosses"]
            for record in records
        ]
        hypotheses = [
            record["hypothesis_glosses"]
            for record in records
        ]
        normalized_references = normalize_token_sequences(
            references,
            args.preprocess_script,
        )
        normalized_hypotheses = normalize_token_sequences(
            hypotheses,
            args.preprocess_script,
        )
        signers = sorted({
            str(record.get("signer", ""))
            for record in records
        })
        per_signer = {}
        for signer in signers:
            indices = [
                index
                for index, record in enumerate(records)
                if str(record.get("signer", "")) == signer
            ]
            per_signer[signer] = aggregate_wer(
                [normalized_references[index] for index in indices],
                [normalized_hypotheses[index] for index in indices],
            )
        results[str(Path(path).expanduser().resolve())] = {
            "overall": aggregate_wer(
                normalized_references,
                normalized_hypotheses,
            ),
            "per_signer": per_signer,
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
