import math

import pytest

from knowledge_aug_lab.models import Chunk
from knowledge_aug_lab.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever

CHUNKS = [
    Chunk("a#0", "a", "alpha beta", 0, 10),
    Chunk("b#0", "b", "beta gamma", 0, 10),
]


@pytest.mark.parametrize("query", ["", "   ", None, 1])
@pytest.mark.parametrize("retriever", [BM25Retriever().fit(CHUNKS), TfidfRetriever().fit(CHUNKS)])
def test_retrievers_reject_empty_or_non_string_queries(retriever: object, query: object) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.retrieve(query)  # type: ignore[attr-defined]


@pytest.mark.parametrize("top_k", [True, False, 0, -1, 1.5, None])
@pytest.mark.parametrize("retriever", [BM25Retriever().fit(CHUNKS), TfidfRetriever().fit(CHUNKS)])
def test_retrievers_reject_invalid_top_k(retriever: object, top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("alpha", top_k=top_k)  # type: ignore[attr-defined]


@pytest.mark.parametrize("k1", [True, "1", -0.1, math.nan, math.inf, -math.inf, 10**400])
def test_bm25_rejects_invalid_k1(k1: object) -> None:
    expected = TypeError if isinstance(k1, (str, bool)) else ValueError
    with pytest.raises(expected):
        BM25Retriever(k1=k1)  # type: ignore[arg-type]


@pytest.mark.parametrize("b", [True, "0.5", -0.1, 1.1, math.nan, math.inf, -math.inf, 10**400])
def test_bm25_rejects_invalid_b(b: object) -> None:
    expected = TypeError if isinstance(b, (str, bool)) else ValueError
    with pytest.raises(expected):
        BM25Retriever(b=b)  # type: ignore[arg-type]


def test_fit_rejects_non_chunk_values() -> None:
    with pytest.raises(TypeError, match="chunks must contain only Chunk values"):
        BM25Retriever().fit([object()])  # type: ignore[list-item]


@pytest.mark.parametrize("rrf_k", [True, -1, 1.5, None])
def test_hybrid_rejects_invalid_rrf_k(rrf_k: object) -> None:
    with pytest.raises(ValueError, match="rrf_k must be a non-negative integer"):
        HybridRetriever([BM25Retriever()], rrf_k=rrf_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("weight", [True, -1.0, 0.0, math.nan, math.inf, -math.inf, "1", 10**400])
def test_hybrid_rejects_invalid_weights(weight: object) -> None:
    with pytest.raises(ValueError, match="weights must be finite positive numbers"):
        HybridRetriever([BM25Retriever()], weights=[weight])  # type: ignore[list-item]


def test_hybrid_validates_configuration_and_request_boundaries() -> None:
    with pytest.raises(ValueError, match="at least one retriever is required"):
        HybridRetriever([])
    with pytest.raises(ValueError, match="weights must match retrievers"):
        HybridRetriever([BM25Retriever()], weights=[])

    hybrid = HybridRetriever([BM25Retriever().fit(CHUNKS)])
    with pytest.raises(ValueError, match="query cannot be empty"):
        hybrid.retrieve(" ")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        hybrid.retrieve("alpha", top_k=True)


def test_empty_corpus_and_refit_have_explicit_behavior() -> None:
    retriever = BM25Retriever().fit(CHUNKS)
    assert retriever.retrieve("alpha", top_k=2)

    retriever.fit([])

    assert retriever.retrieve("alpha", top_k=2) == []


@pytest.mark.parametrize("k1", [math.ulp(0.0), 1e308])
def test_bm25_remains_finite_for_extreme_finite_k1(k1: float) -> None:
    chunk = Chunk("dense#0", "dense", " ".join(["alpha"] * 100), 0, 599)

    results = BM25Retriever(k1=k1, b=1.0).fit([chunk]).retrieve("alpha")

    assert len(results) == 1
    assert math.isfinite(results[0].score)


def test_hybrid_rejects_arithmetic_overflow_configs() -> None:
    with pytest.raises(ValueError, match="rrf_k must fit finite floating-point arithmetic"):
        HybridRetriever([BM25Retriever()], rrf_k=10**400)
    with pytest.raises(ValueError, match="combined weights must produce finite fusion scores"):
        HybridRetriever([BM25Retriever(), BM25Retriever()], weights=[1e308, 1e308], rrf_k=0)
