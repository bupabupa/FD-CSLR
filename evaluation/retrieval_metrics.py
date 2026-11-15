import numpy as np
import torch
import torch.nn.functional as F

from evaluation.frequency_metrics import frequency_bins


def _normalize(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    return F.normalize(values, dim=-1, eps=1e-8)


def retrieval_metrics(
    query_embeddings,
    candidate_embeddings,
    query_ids,
    candidate_ids,
    k_values=(1, 5),
):
    """One-to-many class retrieval with the best matching candidate rank."""
    queries = _normalize(query_embeddings)
    candidates = _normalize(candidate_embeddings)
    query_ids = torch.as_tensor(query_ids, dtype=torch.long).reshape(-1)
    candidate_ids = torch.as_tensor(candidate_ids, dtype=torch.long).reshape(-1)
    if queries.size(0) != query_ids.numel():
        raise ValueError("query ids must match query embeddings")
    if candidates.size(0) != candidate_ids.numel():
        raise ValueError("candidate ids must match candidate embeddings")
    if queries.size(1) != candidates.size(1):
        raise ValueError("query and candidate dimensions must match")
    if queries.size(0) == 0 or candidates.size(0) == 0:
        raise ValueError("retrieval requires at least one query and candidate")
    valid_candidates = candidate_ids >= 0
    candidates = candidates[valid_candidates]
    candidate_ids = candidate_ids[valid_candidates]
    if candidates.size(0) == 0:
        raise ValueError("retrieval has no valid candidates")

    similarity = queries @ candidates.transpose(0, 1)
    order = similarity.argsort(dim=-1, descending=True)
    ranks = []
    for row, query_id in enumerate(query_ids.tolist()):
        matching = (candidate_ids == int(query_id)).nonzero(as_tuple=False)
        if matching.numel() == 0:
            continue
        positions = []
        for candidate_index in matching.reshape(-1).tolist():
            positions.append(
                int((order[row] == int(candidate_index)).nonzero().item())
            )
        ranks.append(min(positions) + 1)
    if not ranks:
        raise ValueError("no query has a valid matching candidate")
    rank_tensor = torch.tensor(ranks, dtype=torch.float32)
    result = {
        "num_queries": len(ranks),
        "mean_rank": float(rank_tensor.mean().item()),
        "mrr": float((1.0 / rank_tensor).mean().item()),
    }
    for k in sorted({int(value) for value in k_values}):
        if k < 1:
            raise ValueError("retrieval k must be positive")
        result["recall@{}".format(k)] = float(
            (rank_tensor <= k).to(torch.float32).mean().item()
        )
    return result


def stratified_retrieval_metrics(
    query_embeddings,
    candidate_embeddings,
    support,
    train_counts,
    fractions=(0.2, 0.3, 0.5),
    k_values=(1, 5),
):
    """Evaluate supported non-blank class ids in head/middle/tail bins."""
    support = torch.as_tensor(support, dtype=torch.float32).reshape(-1)
    train_counts = np.asarray(train_counts, dtype=np.int64).reshape(-1)
    if support.numel() != train_counts.size:
        raise ValueError("support and training counts must have equal length")
    supported_ids = (
        (torch.arange(support.numel(), device=support.device) > 0)
        & (support > 0)
    )
    ids = supported_ids.nonzero(as_tuple=False).reshape(-1)
    class_ids = ids.tolist()
    bins = frequency_bins(train_counts[1:], fractions=fractions)
    candidate_ids = ids
    output = {
        "overall": retrieval_metrics(
            query_embeddings[ids],
            candidate_embeddings[candidate_ids],
            ids,
            candidate_ids,
            k_values=k_values,
        ),
        "bins": {},
    }
    for name, local_ids in bins.items():
        # Bins index non-blank classes, so shift their ids by one.
        selected = [int(value) + 1 for value in local_ids]
        selected = [value for value in selected if value in class_ids]
        if not selected:
            continue
        selected_tensor = torch.tensor(
            selected,
            dtype=torch.long,
            device=query_embeddings.device,
        )
        output["bins"][name] = retrieval_metrics(
            query_embeddings[selected_tensor],
            candidate_embeddings[candidate_ids],
            selected_tensor,
            candidate_ids,
            k_values=k_values,
        )
    return output
