from knowledge_aug_lab.augmentation import ContextCache, MemoryStore, TableStore, ToolRegistry
from knowledge_aug_lab.models import Document


def test_cag_preloads_corpus_once_and_reuses_the_same_context() -> None:
    cache = ContextCache(
        [
            Document("rag", "RAG performs retrieval for each query."),
            Document("cag", "CAG reuses a preloaded bounded context for low latency."),
        ],
        max_characters=500,
    )

    first = cache.context_for("What does CAG reuse?")
    second = cache.context_for("Why can CAG be fast?")

    assert "preloaded bounded context" in first
    assert first == second
    assert cache.build_count == 1
    assert cache.hit_count == 2


def test_table_augmented_generation_executes_typed_aggregation() -> None:
    table = TableStore(
        [
            {"strategy": "RAG", "latency_ms": 120},
            {"strategy": "RAG", "latency_ms": 180},
            {"strategy": "CAG", "latency_ms": 35},
        ]
    )

    result = table.aggregate("latency_ms", operation="mean", where={"strategy": "RAG"})

    assert result.value == 150
    assert result.rows_used == 2
    assert result.provenance == [0, 1]


def test_memory_augmented_generation_recalls_relevant_session_facts() -> None:
    memory = MemoryStore()
    memory.remember("user-7", "The user prefers local models and no cloud APIs.")
    memory.remember("user-7", "The current project demonstrates knowledge augmentation.")
    memory.remember("other-user", "The user prefers hosted models.")

    recalled = memory.recall("user-7", "Which model deployment preference should I follow?", top_k=1)

    assert recalled == ["The user prefers local models and no cloud APIs."]


def test_memory_recall_rejects_invalid_limits_and_excludes_zero_overlap() -> None:
    memory = MemoryStore()
    memory.remember("user-7", "The user prefers local models.")

    assert memory.recall("user-7", "unrelated weather forecast", top_k=1) == []

    try:
        memory.recall("user-7", "models", top_k=0)
    except ValueError as error:
        assert str(error) == "top_k must be positive"
    else:
        raise AssertionError("top_k=0 must be rejected")


def test_tool_augmented_generation_calls_allowlisted_tool_with_structured_arguments() -> None:
    tools = ToolRegistry()
    tools.register("estimate_cost", lambda tokens, rate: tokens / 1_000_000 * rate)

    result = tools.call("estimate_cost", tokens=2_500_000, rate=0.40)

    assert result.name == "estimate_cost"
    assert result.output == 1.0
    assert result.arguments == {"tokens": 2_500_000, "rate": 0.40}
