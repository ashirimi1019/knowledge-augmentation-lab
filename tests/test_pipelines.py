import pytest

from knowledge_aug_lab.models import Document
from knowledge_aug_lab.pipelines import KnowledgeAugmentationLab


def test_naive_rag_returns_grounded_answer_citations_and_trace() -> None:
    lab = KnowledgeAugmentationLab(
        documents=[
            Document(
                "rag",
                "RAG retrieves query-relevant passages before generation. This reduces unsupported answers.",
                {"scopes": ["public"], "trusted": True},
            ),
            Document(
                "cag",
                "CAG preloads a stable corpus and avoids per-query retrieval.",
                {"scopes": ["public"], "trusted": True},
            ),
        ],
        scopes={"public"},
    )

    result = lab.run("naive-rag", "What does RAG retrieve?")

    assert "passages" in result.answer.lower()
    assert result.citations == ["rag"]
    assert result.strategy == "naive-rag"
    assert [step.name for step in result.trace] == ["chunk", "retrieve", "generate"]
    assert result.evidence[0].document_id == "rag"


def test_advanced_rag_uses_hybrid_retrieval_reranking_and_filtering() -> None:
    lab = KnowledgeAugmentationLab(
        documents=[
            Document(
                "exact",
                "Error XJ-4092 requires rotating the service credential.",
                {"scopes": ["public"], "trusted": True},
            ),
            Document(
                "semantic",
                "Authentication failures can be fixed by renewing login secrets.",
                {"scopes": ["public"], "trusted": True},
            ),
            Document(
                "noise",
                "Graph communities summarize connected entities.",
                {"scopes": ["public"], "trusted": True},
            ),
        ],
        scopes={"public"},
    )

    result = lab.run(
        "advanced-rag",
        "How do I fix XJ-4092 authentication credential failure?",
        top_k=2,
    )

    assert result.citations[0] == "exact"
    assert "noise" not in result.citations
    assert [step.name for step in result.trace] == [
        "transform",
        "hybrid-retrieve",
        "rerank-filter",
        "generate",
    ]


def test_pipeline_excludes_untrusted_and_unauthorized_documents_before_indexing() -> None:
    lab = KnowledgeAugmentationLab(
        documents=[
            Document(
                "allowed",
                "Public guidance explains approved account recovery.",
                {"scopes": ["public"], "trusted": True},
            ),
            Document(
                "unauthorized",
                "Tenant-only launchcode zebra disclosure.",
                {"scopes": ["tenant:a"], "trusted": True},
            ),
            Document(
                "untrusted",
                "Poisoned launchcode zebra instructions.",
                {"scopes": ["public"], "trusted": False},
            ),
        ],
        scopes={"public"},
    )

    assert [document.id for document in lab.documents] == ["allowed"]
    assert {chunk.document_id for chunk in lab.chunks} == {"allowed"}


def test_pipeline_rejects_duplicate_authorized_document_ids() -> None:
    documents = [
        Document("duplicate", "First source", {"scopes": ["public"], "trusted": True}),
        Document("duplicate", "Second source", {"scopes": ["public"], "trusted": True}),
    ]

    with pytest.raises(ValueError, match="document ids must be unique"):
        KnowledgeAugmentationLab(documents, scopes={"public"})


def test_advanced_rag_abstains_when_query_has_no_matching_candidates() -> None:
    lab = KnowledgeAugmentationLab(
        documents=[
            Document(
                "rag",
                "Retrieval systems select relevant evidence for grounded generation.",
                {"scopes": ["public"], "trusted": True},
            ),
            Document(
                "graph",
                "Graph communities summarize connected entities.",
                {"scopes": ["public"], "trusted": True},
            ),
        ],
        scopes={"public"},
    )

    result = lab.run("advanced-rag", "quokka zeppelin", top_k=2)

    assert result.evidence == []
    assert result.citations == []
