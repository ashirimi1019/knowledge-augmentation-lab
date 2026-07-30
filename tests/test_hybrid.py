from knowledge_aug_lab.models import Document
from knowledge_aug_lab.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever
from knowledge_aug_lab.text import RecursiveChunker


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
