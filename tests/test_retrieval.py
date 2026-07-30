import pytest

from knowledge_aug_lab.models import Chunk, Document
from knowledge_aug_lab.retrieval import BM25Retriever, TfidfRetriever
from knowledge_aug_lab.text import RecursiveChunker


def test_document_to_ranked_context_is_end_to_end() -> None:
    documents = [
        Document(
            id="rag",
            text=(
                "Retrieval augmented generation retrieves relevant evidence before generation. "
                "It grounds answers in external documents and can provide citations."
            ),
            metadata={"topic": "RAG"},
        ),
        Document(
            id="cag",
            text=(
                "Cache augmented generation preloads a bounded stable corpus into a model context "
                "or reusable key value cache."
            ),
            metadata={"topic": "CAG"},
        ),
    ]

    chunks = RecursiveChunker(chunk_size=120, overlap=20).split(documents)
    results = BM25Retriever().fit(chunks).retrieve("How does retrieval ground an answer?", top_k=2)

    assert chunks
    assert results[0].chunk.document_id == "rag"
    assert results[0].score > results[1].score
    assert results[0].rank == 1


def test_chunk_spans_match_trimmed_text_exactly() -> None:
    document = Document("spans", "  Grounded evidence.  ")

    chunk = RecursiveChunker().split([document])[0]

    assert chunk.text == "Grounded evidence."
    assert document.text[chunk.start : chunk.end] == chunk.text


def test_duplicate_document_and_chunk_ids_are_rejected() -> None:
    duplicate_documents = [Document("same", "First source"), Document("same", "Second source")]

    with pytest.raises(ValueError, match="document ids must be unique"):
        RecursiveChunker().split(duplicate_documents)

    duplicate_chunks = [
        Chunk("same#0", "first", "First source", 0, 12),
        Chunk("same#0", "second", "Second source", 0, 13),
    ]
    for retriever in (BM25Retriever(), TfidfRetriever()):
        with pytest.raises(ValueError, match="chunk ids must be unique"):
            retriever.fit(duplicate_chunks)
