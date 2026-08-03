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


@pytest.mark.parametrize("query", ["How is storage managed?", "What stage runs next?", "Open the cage."])
def test_graph_entity_matching_rejects_substring_false_positives(query: str) -> None:
    graph = KnowledgeGraph([("RAG", "contrasts", "TAG"), ("CAG", "uses", "Context cache")])

    assert graph.match_entities(query) == []


def test_graph_entity_matching_supports_exact_terms_and_contiguous_phrases() -> None:
    graph = KnowledgeGraph([("RAG", "uses", "Context cache")])

    assert graph.match_entities("Compare rag with a context cache") == ["RAG", "Context cache"]
    assert graph.match_entities("Compare context reusable cache") == []


def test_graph_entity_matching_uses_identical_unicode_casefold_normalization() -> None:
    graph = KnowledgeGraph([("Straße", "connects", "Knowledge")])

    assert graph.match_entities("Explain Straße") == ["Straße"]
    assert graph.match_entities("Explain STRASSE") == ["Straße"]
