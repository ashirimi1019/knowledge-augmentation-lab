import pytest

from knowledge_aug_lab.knowledge import KnowledgeGraph


@pytest.mark.parametrize(
    ("subject", "predicate", "object_", "field"),
    [
        ("", "uses", "Retrieval", "subject"),
        ("RAG", " ", "Retrieval", "predicate"),
        ("RAG", "uses", None, "object"),
    ],
)
def test_graph_rejects_blank_or_non_string_triple_values(
    subject: object,
    predicate: object,
    object_: object,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        KnowledgeGraph().add(subject, predicate, object_)  # type: ignore[arg-type]


def test_duplicate_facts_are_suppressed_in_traversal_and_reports() -> None:
    graph = KnowledgeGraph()
    graph.add(" RAG ", " uses ", " Retrieval ")
    graph.add("RAG", "uses", "Retrieval")
    graph.add("rag", "USES", "retrieval")

    assert [fact.render() for fact in graph.neighborhood("rag")] == ["RAG --uses--> Retrieval"]
    assert graph.community_reports()[0].summary == "RAG --uses--> Retrieval"


@pytest.mark.parametrize("max_hops", [True, 1.5, "2", None])
def test_graph_max_hops_must_be_an_integer(max_hops: object) -> None:
    graph = KnowledgeGraph([("RAG", "uses", "Retrieval")])
    with pytest.raises(TypeError, match="max_hops must be an integer"):
        graph.neighborhood("RAG", max_hops=max_hops)  # type: ignore[arg-type]


def test_non_positive_hops_return_no_facts() -> None:
    graph = KnowledgeGraph([("RAG", "uses", "Retrieval")])

    assert graph.neighborhood("RAG", max_hops=0) == []
    assert graph.neighborhood("RAG", max_hops=-1) == []
