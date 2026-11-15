import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_metrics import stratified_retrieval_metrics


def _state_dict(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    return payload


def _strip_prefixes(state):
    prefixes = ("module.", "model.")
    output = {}
    for key, value in state.items():
        while key.startswith(prefixes):
            key = key.split(".", 1)[1]
        output[key] = value
    return output


def load_prototype_retrieval_inputs(checkpoint_path, train_counts_path):
    state = _strip_prefixes(_state_dict(checkpoint_path))
    visual = state.get("prototype_head.prototypes")
    support = state.get("prototype_head.support")
    embeddings = state.get("text_prototype_head.lexical_embeddings")
    projection = state.get("text_prototype_head.projection.weight")
    if visual is None or support is None:
        raise ValueError("checkpoint does not contain visual prototypes")
    if embeddings is None or projection is None:
        raise ValueError("checkpoint does not contain a text prototype head")
    text = F.normalize(
        torch.as_tensor(embeddings, dtype=torch.float32)
        @ torch.as_tensor(projection, dtype=torch.float32).t(),
        dim=-1,
        eps=1e-8,
    )
    train_counts = np.load(train_counts_path).astype(np.int64).reshape(-1)
    if train_counts.size != visual.size(0):
        raise ValueError("training counts must match the class count")
    return {
        "visual": torch.as_tensor(visual, dtype=torch.float32),
        "text": text,
        "support": torch.as_tensor(support, dtype=torch.float32),
        "train_counts": train_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-counts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5])
    args = parser.parse_args()
    inputs = load_prototype_retrieval_inputs(
        args.checkpoint,
        args.train_counts,
    )
    result = stratified_retrieval_metrics(
        inputs["visual"],
        inputs["text"],
        inputs["support"],
        inputs["train_counts"],
        k_values=args.k,
    )
    result["checkpoint"] = str(Path(args.checkpoint).resolve())
    result["train_counts"] = str(Path(args.train_counts).resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
