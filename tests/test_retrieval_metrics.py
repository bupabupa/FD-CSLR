import torch

from evaluation.retrieval_metrics import (
    retrieval_metrics,
    stratified_retrieval_metrics,
)


def test_retrieval_recall_and_mrr():
    queries = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    candidates = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ]
    )
    result = retrieval_metrics(
        queries,
        candidates,
        query_ids=[1, 2],
        candidate_ids=[1, 2, 1],
        k_values=(1, 2),
    )
    assert result["recall@1"] == 1.0
    assert result["recall@2"] == 1.0
    assert result["mrr"] == 1.0


def test_stratified_retrieval_uses_supported_nonblank_classes():
    visual = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ]
    )
    text = visual.clone()
    support = torch.tensor([100.0, 4.0, 2.0, 1.0])
    counts = torch.tensor([0, 40, 10, 1])
    result = stratified_retrieval_metrics(
        visual,
        text,
        support,
        counts,
        k_values=(1,),
    )
    assert result["overall"]["recall@1"] == 1.0
    assert set(result["bins"]) == {"head", "middle", "tail"}


def test_stratified_retrieval_keeps_the_full_candidate_pool():
    visual = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    text = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ]
    )
    result = stratified_retrieval_metrics(
        visual,
        text,
        support=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        train_counts=torch.tensor([0, 40, 10, 1]),
        k_values=(1,),
    )
    # Tail class 3 is retrieved behind head class 1 from the same global pool.
    assert result["bins"]["tail"]["recall@1"] == 0.0
    assert result["bins"]["middle"]["recall@1"] == 1.0


if __name__ == "__main__":
    test_retrieval_recall_and_mrr()
    test_stratified_retrieval_uses_supported_nonblank_classes()
    test_stratified_retrieval_keeps_the_full_candidate_pool()
    print("retrieval metric tests passed")
