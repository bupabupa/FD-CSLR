import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.signer_protocol import (  # noqa: E402
    assign_pseudo_signers,
    cluster_pseudo_signers,
    save_pseudo_signers,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--assign-features",
        default=None,
        help=(
            "optional npz with fileids/embeddings for cal, sel and test; "
            "records are assigned to training-fitted centers only"
        ),
    )
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mapping-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--centers-output", default=None)
    args = parser.parse_args()

    with np.load(args.features, allow_pickle=False) as source:
        if "fileids" not in source.files or "embeddings" not in source.files:
            raise ValueError("features npz requires fileids and embeddings")
        fileids = source["fileids"].tolist()
        embeddings = source["embeddings"]
    result = cluster_pseudo_signers(
        fileids,
        embeddings,
        clusters=args.clusters,
        seeds=args.seeds,
        iterations=args.iterations,
    )
    report = dict(result["report"])
    report["fit_num_samples"] = int(report["num_samples"])
    report["fit_fileids_sha256"] = hashlib.sha256(
        json.dumps(sorted(result["mapping"]), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if args.assign_features:
        with np.load(args.assign_features, allow_pickle=False) as source:
            if "fileids" not in source.files or "embeddings" not in source.files:
                raise ValueError(
                    "assign features npz requires fileids and embeddings"
                )
            assign_fileids = source["fileids"].tolist()
            assign_embeddings = source["embeddings"]
        assigned = assign_pseudo_signers(
            assign_fileids,
            assign_embeddings,
            result["centers"],
        )
        overlap = set(result["mapping"]).intersection(assigned["mapping"])
        if overlap:
            raise ValueError(
                "fit and assign feature files overlap on fileids: {}".format(
                    sorted(overlap)[:8]
                )
            )
        result["mapping"].update(assigned["mapping"])
        report["assign_num_samples"] = assigned["report"]["num_samples"]
        report["assign_cluster_sizes"] = assigned["report"]["cluster_sizes"]
        report["assign_features"] = str(
            Path(args.assign_features).resolve()
        )
    report["num_samples"] = len(result["mapping"])
    report["cluster_sizes"] = np.bincount(
        np.asarray(list(result["mapping"].values()), dtype=np.int64),
        minlength=int(result["report"]["num_clusters"]),
    ).tolist()
    result["report"] = report
    if args.centers_output:
        centers_path = Path(args.centers_output)
        centers_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(centers_path, np.asarray(result["centers"], dtype=np.float32))
        report["centers_output"] = str(centers_path.resolve())
    save_pseudo_signers(
        result,
        args.mapping_output,
        args.report_output,
    )
    print(
        "pseudo signer map: {} clusters, {} records, mean ARI {:.6f}".format(
            result["report"]["num_clusters"],
            result["report"]["num_samples"],
            result["report"]["mean_pairwise_ari"],
        )
    )


if __name__ == "__main__":
    main()
