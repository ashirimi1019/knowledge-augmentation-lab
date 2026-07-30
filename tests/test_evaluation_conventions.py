import math

import pytest

from knowledge_aug_lab.evaluation import evaluate_retrieval, ndcg_at_k


def test_retrieval_metrics_use_returned_result_precision() -> None:
    metrics = evaluate_retrieval(["rag"], {"rag", "graph"}, k=3)

    assert metrics.recall_at_k == 0.5
    assert metrics.reciprocal_rank == 1.0
    assert metrics.precision_at_k == 1.0
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.retrieved_count == 1


def test_ndcg_rewards_earlier_relevant_results_and_deduplicates_ids() -> None:
    ideal = ndcg_at_k(["rag", "rag", "graph"], {"rag", "graph"}, k=3)
    delayed = ndcg_at_k(["noise", "rag", "graph"], {"rag", "graph"}, k=3)

    assert ideal == 1.0
    assert 0.0 < delayed < ideal


@pytest.mark.parametrize("k", [True, 0, -1, 1.5, None])
def test_evaluation_k_must_be_a_positive_integer(k: object) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluate_retrieval([], set(), k=k)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must be a positive integer"):
        ndcg_at_k([], set(), k=k)  # type: ignore[arg-type]


def test_ndcg_is_bounded_for_partial_relevance() -> None:
    score = ndcg_at_k(["noise", "rag", "other"], {"rag", "graph"}, k=3)

    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
