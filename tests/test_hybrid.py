import pytest

from knowledge_aug_lab.models import Chunk, Document, RetrievalResult
from knowledge_aug_lab.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever
from knowledge_aug_lab.text import RecursiveChunker


class StubRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return list(self.results)


def test_hybrid_rrf_combines_sparse_and_vector_space_rankings() -> None:
    chunks = RecursiveChunker(chunk_size=200, overlap=0).split(
        [
            Document("exact", "Incident XJ-4092 is fixed by rotating the service credential."),
            Document("semantic", "Authentication failures require renewing the login secret."),
            Document("noise", "Vector indexes trade memory for nearest-neighbor search speed."),
        ]
    )
    hybrid = HybridRetriever(
        retrievers=[BM25Retriever().fit(chunks), TfidfRetriever().fit(chunks)],
        weights=[1.0, 1.0],
    )

    results = hybrid.retrieve("XJ-4092 authentication credential failure", top_k=3)

    assert {result.chunk.document_id for result in results} == {"exact", "semantic"}
    assert all(result.retriever == "hybrid-rrf" for result in results)
    assert [result.rank for result in results] == [1, 2]


def test_hybrid_rejects_conflicting_chunks_with_the_same_id() -> None:
    first = Chunk("shared#0", "first", "alpha from first source", 0, 23)
    second = Chunk("shared#0", "second", "alpha from second source", 0, 24)
    hybrid = HybridRetriever(
        retrievers=[BM25Retriever().fit([first]), BM25Retriever().fit([second])],
    )

    with pytest.raises(ValueError, match="conflicting chunks share id 'shared#0'"):
        hybrid.retrieve("alpha")


def test_hybrid_rejects_duplicate_chunk_ids_within_one_ranking() -> None:
    chunk = Chunk("shared#0", "shared", "alpha evidence", 0, 14)
    retriever = StubRetriever(
        [
            RetrievalResult(chunk, 2.0, 1, "stub"),
            RetrievalResult(chunk, 1.0, 2, "stub"),
        ]
    )

    with pytest.raises(ValueError, match="retriever returned duplicate chunk id: 'shared#0'"):
        HybridRetriever([retriever], rrf_k=0).retrieve("alpha")


@pytest.mark.parametrize("ranks", [(1, 1), (2, 1), (1, 3)])
def test_hybrid_requires_contiguous_ordered_source_ranks(ranks: tuple[int, int]) -> None:
    first = Chunk("first#0", "first", "alpha one", 0, 9)
    second = Chunk("second#0", "second", "alpha two", 0, 9)
    retriever = StubRetriever(
        [
            RetrievalResult(first, 2.0, ranks[0], "stub"),
            RetrievalResult(second, 1.0, ranks[1], "stub"),
        ]
    )

    with pytest.raises(ValueError, match="retriever ranks must be contiguous and ordered from 1"):
        HybridRetriever([retriever]).retrieve("alpha")
