import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


FIELDS = {
    "support": "prototype_head.support",
    "prototypes": "prototype_head.prototypes",
    "prototype_gate": "prototype_gate",
    "fdgc_difficulty": "fdgc_difficulty",
    "variances": "prototype_head.variances",
}


def _model_state(checkpoint):
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dictionary")
    return state


def _find_field(state, suffix):
    matches = [
        key for key in state
        if key == suffix or key.endswith("." + suffix)
    ]
    if len(matches) > 1:
        raise ValueError("ambiguous checkpoint field: {}".format(suffix))
    return state[matches[0]] if matches else None


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    quantiles = np.percentile(values, [0, 25, 50, 75, 90, 99, 100])
    return {
        "count": int(values.size),
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            label: float(value)
            for label, value in zip(
                ["min", "p25", "p50", "p75", "p90", "p99", "max"],
                quantiles,
            )
        },
    }


def export_coverage(checkpoint_path, min_support=0.05):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = _model_state(checkpoint)
    support = _find_field(state, FIELDS["support"])
    prototypes = _find_field(state, FIELDS["prototypes"])
    if support is None or prototypes is None:
        raise ValueError("checkpoint does not contain prototype statistics")
    support = support.detach().float().reshape(-1).cpu()
    prototypes = prototypes.detach().float().cpu()
    if support.numel() != prototypes.size(0):
        raise ValueError("prototype support and prototype rows disagree")

    established = support >= float(min_support)
    norms = prototypes.norm(dim=-1)
    output = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "split_sha256": checkpoint.get("split_sha256"),
        "min_support": float(min_support),
        "num_classes": int(support.numel()),
        "established_classes": int(established.sum()),
        "coverage_fraction": float(established.float().mean()),
        "support_summary": _summary(support),
        "established_support_summary": _summary(
            support[established]
        ),
        "prototype_norm_summary": _summary(norms),
        "classes": [
            {
                "class_id": int(index),
                "support": float(support[index]),
                "established": bool(established[index]),
                "prototype_norm": float(norms[index]),
            }
            for index in range(support.numel())
        ],
    }

    gate = _find_field(state, FIELDS["prototype_gate"])
    if gate is not None:
        gate_value = float(gate.detach().float().reshape(-1)[0].cpu())
        output["prototype_gate_logit"] = gate_value
        output["prototype_gate_probability"] = float(
            1.0 / (1.0 + math.exp(-gate_value))
        )

    difficulty = _find_field(state, FIELDS["fdgc_difficulty"])
    if difficulty is not None:
        difficulty = difficulty.detach().float().reshape(-1).cpu()
        output["fdgc_difficulty_summary"] = _summary(difficulty)
        output["fdgc_difficulty_nonzero_classes"] = int(
            (difficulty > 0).sum()
        )

    variances = _find_field(state, FIELDS["variances"])
    if variances is not None:
        output["variance_summary"] = _summary(
            variances.detach().float().reshape(-1).cpu()
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-support", type=float, default=0.05)
    args = parser.parse_args()
    result = export_coverage(args.checkpoint, args.min_support)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "checkpoint": result["checkpoint"],
        "num_classes": result["num_classes"],
        "established_classes": result["established_classes"],
        "coverage_fraction": result["coverage_fraction"],
        "prototype_gate_probability": result.get(
            "prototype_gate_probability"
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
