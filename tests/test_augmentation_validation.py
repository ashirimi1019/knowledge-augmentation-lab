import pytest

from knowledge_aug_lab.augmentation import ContextCache, MemoryStore, ToolSpec
from knowledge_aug_lab.models import Document


@pytest.mark.parametrize("documents", [None, {}, ["not-a-document"]])
def test_context_cache_validates_documents(documents: object) -> None:
    with pytest.raises(TypeError, match="documents must be a list of Document values"):
        ContextCache(documents)  # type: ignore[arg-type]


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, None])
def test_context_cache_validates_budget(budget: object) -> None:
    with pytest.raises(ValueError, match="max_characters must be a positive integer"):
        ContextCache([Document("doc", "text")], max_characters=budget)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", " ", None, 1])
def test_context_cache_validates_query(query: object) -> None:
    cache = ContextCache([])
    with pytest.raises(ValueError, match="query cannot be empty"):
        cache.context_for(query)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("scope", "fact"),
    [("", "fact"), ("scope", " "), (None, "fact"), ("scope", None)],
)
def test_memory_remember_validates_scope_and_fact(scope: object, fact: object) -> None:
    with pytest.raises(ValueError, match="scope and fact are required"):
        MemoryStore().remember(scope, fact)  # type: ignore[arg-type]


def test_memory_normalizes_scope_and_suppresses_duplicate_facts() -> None:
    memory = MemoryStore()
    memory.remember(" user ", " prefers local models ")
    memory.remember("user", "prefers local models")

    assert memory.recall("user", "local models", top_k=3) == ["prefers local models"]


@pytest.mark.parametrize("scope", ["", " ", None, 1])
def test_memory_recall_validates_scope(scope: object) -> None:
    with pytest.raises(ValueError, match="scope cannot be empty"):
        MemoryStore().recall(scope, "query")  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", " ", None, 1])
def test_memory_recall_validates_query(query: object) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        MemoryStore().recall("scope", query)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [True, False, 0, -1, 1.5, None])
def test_memory_recall_validates_top_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        MemoryStore().recall("scope", "query", top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "bad-name", "description": "desc", "function": lambda: None, "allowed_arguments": frozenset()},
        {"name": "good", "description": " ", "function": lambda: None, "allowed_arguments": frozenset()},
        {"name": "good", "description": "desc", "function": 1, "allowed_arguments": frozenset()},
        {
            "name": "good",
            "description": "desc",
            "function": lambda: None,
            "allowed_arguments": frozenset({"bad-arg"}),
        },
        {
            "name": "good",
            "description": "desc",
            "function": lambda: None,
            "allowed_arguments": frozenset({"unexpected"}),
        },
    ],
)
def test_tool_spec_validates_its_public_contract(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ToolSpec(**kwargs)  # type: ignore[arg-type]
