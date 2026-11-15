import json
from pathlib import Path

import numpy as np


def load_signer_map(path):
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("signer map must be a non-empty JSON object")
    output = {}
    for fileid, signer_id in payload.items():
        value = int(signer_id)
        if value < 0:
            raise ValueError("pseudo signer ids must be non-negative")
        output[str(fileid)] = value
    if len(set(output.values())) < 2:
        raise ValueError("pseudo signer map must contain at least two clusters")
    return output


def adjusted_rand_index(first, second):
    first = np.asarray(first, dtype=np.int64).reshape(-1)
    second = np.asarray(second, dtype=np.int64).reshape(-1)
    if first.size != second.size or first.size == 0:
        raise ValueError("ARI inputs must be non-empty and equal length")

    def choose2(values):
        values = values.astype(np.float64)
        return float(np.sum(values * (values - 1.0) / 2.0))

    _, first_inverse, first_counts = np.unique(
        first,
        return_inverse=True,
        return_counts=True,
    )
    _, second_inverse, second_counts = np.unique(
        second,
        return_inverse=True,
        return_counts=True,
    )
    contingency = np.zeros(
        (first_counts.size, second_counts.size),
        dtype=np.int64,
    )
    np.add.at(contingency, (first_inverse, second_inverse), 1)
    sum_comb = choose2(contingency)
    first_comb = choose2(first_counts)
    second_comb = choose2(second_counts)
    total_comb = choose2(np.array([first.size]))
    expected = first_comb * second_comb / max(total_comb, 1e-12)
    maximum = 0.5 * (first_comb + second_comb)
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0
    return float((sum_comb - expected) / denominator)


def _kmeans_once(embeddings, clusters, seed, iterations):
    rng = np.random.default_rng(int(seed))
    embeddings = np.asarray(embeddings, dtype=np.float64)
    count = embeddings.shape[0]
    if clusters < 2 or clusters > count:
        raise ValueError("cluster count must be between two and sample count")
    center_ids = [int(rng.integers(0, count))]
    squared = np.sum(
        (embeddings - embeddings[center_ids[0]]) ** 2,
        axis=1,
    )
    for _ in range(1, clusters):
        probabilities = squared / max(float(squared.sum()), 1e-12)
        center_ids.append(int(rng.choice(count, p=probabilities)))
        distance = np.sum(
            (embeddings - embeddings[center_ids[-1]]) ** 2,
            axis=1,
        )
        squared = np.minimum(squared, distance)
    centers = embeddings[center_ids].copy()
    labels = np.full(count, -1, dtype=np.int64)
    for _ in range(int(iterations)):
        distances = np.sum(
            (embeddings[:, None, :] - centers[None, :, :]) ** 2,
            axis=-1,
        )
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for cluster in range(clusters):
            members = embeddings[labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
    inertia = float(
        np.sum((embeddings - centers[labels]) ** 2)
    )
    return labels, centers, inertia


def cluster_pseudo_signers(
    fileids,
    embeddings,
    clusters,
    seeds=(0, 1, 2),
    iterations=100,
):
    fileids = [str(value) for value in fileids]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(fileids):
        raise ValueError("embeddings must have shape [N,D] and match fileids")
    if len(set(fileids)) != len(fileids):
        raise ValueError("fileids must be unique")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    seeds = tuple(int(value) for value in seeds)
    if not seeds:
        raise ValueError("at least one clustering seed is required")
    runs = [
        _kmeans_once(embeddings, clusters, seed, iterations)
        for seed in seeds
    ]
    best_index = min(range(len(runs)), key=lambda index: runs[index][2])
    labels = runs[best_index][0]
    stability = [
        adjusted_rand_index(labels, run[0])
        for index, run in enumerate(runs)
        if index != best_index
    ]
    return {
        "mapping": {
            fileid: int(label)
            for fileid, label in zip(fileids, labels.tolist())
        },
        "centers": runs[best_index][1],
        "selected_seed": seeds[best_index],
        "report": {
            "num_samples": len(fileids),
            "num_clusters": int(clusters),
            "seeds": list(seeds),
            "selected_seed": seeds[best_index],
            "inertia": runs[best_index][2],
            "cluster_sizes": np.bincount(
                labels,
                minlength=int(clusters),
            ).tolist(),
            "mean_pairwise_ari": (
                float(np.mean(stability)) if stability else 1.0
            ),
            "min_pairwise_ari": (
                float(np.min(stability)) if stability else 1.0
            ),
        },
    }


def assign_pseudo_signers(fileids, embeddings, centers):
    """Assign unseen records to centroids fitted on the training split."""
    fileids = [str(value) for value in fileids]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(fileids):
        raise ValueError("embeddings must have shape [N,D] and match fileids")
    if len(set(fileids)) != len(fileids):
        raise ValueError("fileids must be unique")
    if centers.ndim != 2 or centers.shape[0] < 2:
        raise ValueError("centers must have shape [K,D] with K >= 2")
    if embeddings.shape[1] != centers.shape[1]:
        raise ValueError("embedding and center dimensions must match")
    if not np.isfinite(embeddings).all() or not np.isfinite(centers).all():
        raise ValueError("embeddings and centers must be finite")
    distances = np.sum(
        (embeddings[:, None, :] - centers[None, :, :]) ** 2,
        axis=-1,
    )
    labels = np.argmin(distances, axis=1)
    return {
        "mapping": {
            fileid: int(label)
            for fileid, label in zip(fileids, labels.tolist())
        },
        "report": {
            "num_samples": len(fileids),
            "num_clusters": int(centers.shape[0]),
            "cluster_sizes": np.bincount(
                labels,
                minlength=int(centers.shape[0]),
            ).tolist(),
        },
    }


def save_pseudo_signers(result, mapping_path, report_path):
    mapping_path = Path(mapping_path)
    report_path = Path(report_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(result["mapping"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(result["report"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
