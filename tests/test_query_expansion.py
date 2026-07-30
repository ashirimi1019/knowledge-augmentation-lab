import pytest

from knowledge_aug_lab.query_expansion import QueryExpander


def test_query_expander_adds_each_unseen_term_once() -> None:
    expander = QueryExpander({"rag": ("retrieval", "generation", "retrieval")})

    assert expander.expand("RAG retrieval") == "RAG retrieval generation"


def test_query_expander_is_configurable() -> None:
    expander = QueryExpander({"outage": ("incident", "failure")})

    assert expander.expand("outage report") == "outage report incident failure"


@pytest.mark.parametrize("query", [None, "", "   "])
def test_query_expander_rejects_empty_queries(query: object) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        QueryExpander().expand(query)  # type: ignore[arg-type]


def test_query_expander_defensively_copies_configuration() -> None:
    configured = ["retrieval"]
    expansions = {"rag": configured}
    expander = QueryExpander(expansions)  # type: ignore[arg-type]
    configured.append("poisoned")

    assert expander.expand("rag") == "rag retrieval"


@pytest.mark.parametrize(
    ("expansions", "expected_exception"),
    [
        ([], TypeError),
        ({"": ("retrieval",)}, ValueError),
        ({"rag": "retrieval"}, TypeError),
        ({"rag": ("",)}, ValueError),
        ({"rag": (1,)}, TypeError),
    ],
)
def test_query_expander_rejects_malformed_configuration(
    expansions: object,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        QueryExpander(expansions)  # type: ignore[arg-type]


def test_query_expander_rejects_keys_that_collide_after_normalization() -> None:
    with pytest.raises(ValueError, match="expansion keys collide after normalization: 'rag'"):
        QueryExpander({"RAG": ("retrieval",), " rag ": ("generation",)})
