import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def metric(payload, section, name):
    value = payload.get(section, {})
    if not value.get("available", False):
        return None
    return value.get(name)


def difference(h1_value, baseline_value):
    if h1_value is None or baseline_value is None:
        return None
    return float(h1_value - baseline_value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--h1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    h1 = load_json(args.h1)
    if baseline.get("split_sha256") != h1.get("split_sha256"):
        raise ValueError("diagnostics use different split manifests")

    metrics = (
        ("accuracy", "signer_probe", "accuracy"),
        ("macro_recall", "signer_probe", "macro_recall"),
        ("nll", "signer_probe", "nll"),
        ("swap_mean_kl", "signer_swap", "mean_kl"),
        ("swap_mean_js", "signer_swap", "mean_js"),
        ("swap_mean_l1", "signer_swap", "mean_l1"),
    )
    comparison = {
        "split_sha256": baseline["split_sha256"],
        "baseline": str(Path(args.baseline).resolve()),
        "h1": str(Path(args.h1).resolve()),
        "metrics": {},
    }
    for key, section, name in metrics:
        baseline_value = metric(baseline, section, name)
        h1_value = metric(h1, section, name)
        comparison["metrics"][key] = {
            "baseline": baseline_value,
            "h1": h1_value,
            "h1_minus_baseline": difference(h1_value, baseline_value),
        }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as destination:
        json.dump(comparison, destination, indent=2, sort_keys=True)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
