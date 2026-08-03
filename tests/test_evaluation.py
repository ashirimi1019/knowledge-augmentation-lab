from knowledge_aug_lab.evaluation import evaluate_retrieval, lexical_groundedness


def test_evaluation_reports_recall_mrr_and_groundedness() -> None:
    retrieval = evaluate_retrieval(
        ranked_document_ids=["noise", "rag", "cag"],
        relevant_document_ids={"rag", "cag"},
        k=2,
    )
    groundedness = lexical_groundedness(
        answer="RAG retrieves evidence and cites sources.",
        context="RAG retrieves relevant evidence. Grounded systems cite their sources.",
    )

    assert retrieval.recall_at_k == 0.5
    assert retrieval.reciprocal_rank == 0.5
    assert groundedness > 0.7


def test_retrieval_metrics_deduplicate_ranked_document_ids() -> None:
    retrieval = evaluate_retrieval(
        ranked_document_ids=["rag", "rag"],
        relevant_document_ids={"rag"},
        k=2,
    )

    assert retrieval.recall_at_k == 1.0
    assert retrieval.precision_at_k == 1.0
    assert retrieval.reciprocal_rank == 1.0


def test_lexical_groundedness_ignores_only_recognized_citation_markers() -> None:
    context = "Evidence sentence."

    assert lexical_groundedness("Evidence sentence. [source-123]", context, citation_ids=["source-123"]) == 1.0
    assert lexical_groundedness("Evidence sentence. [unsupported]", context, citation_ids=["source-123"]) < 1.0
