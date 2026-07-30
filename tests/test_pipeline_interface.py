import pytest

from knowledge_aug_lab.augmentation import MemoryStore, TableStore, ToolRegistry
from knowledge_aug_lab.knowledge import KnowledgeGraph
from knowledge_aug_lab.models import AugmentationResult, Document
from knowledge_aug_lab.pipelines import (
    AugmentationPipeline,
    GraphRAGDemoPipeline,
    HybridRAGPipeline,
    KnowledgeAugmentationLab,
    MemoryAugmentedPipeline,
    NaiveRAGPipeline,
    TableAugmentedPipeline,
    ToolAugmentedPipeline,
)
from knowledge_aug_lab.query_expansion import QueryExpander


def trusted_document() -> Document:
    return Document(
        "rag",
        "Authentication failures can be fixed by renewing login secrets.",
        {"scopes": ["public"], "trusted": True},
    )


class FalseyQueryExpander(QueryExpander):
    def __bool__(self) -> bool:
        return False


def accepts_pipeline(pipeline: AugmentationPipeline) -> AugmentationResult:
    return pipeline.run("authentication failure", top_k=1)


def test_lab_uses_registry_and_injected_query_expander() -> None:
    lab = KnowledgeAugmentationLab(
        [trusted_document()],
        scopes={"public"},
        query_expander=QueryExpander({"identity": ("authentication", "login")}),
    )

    result = lab.run("advanced-rag", "identity failure", top_k=1)

    assert result.strategy == "advanced-rag"
    assert "identity failure authentication login" in result.trace[0].detail
    assert isinstance(lab.pipelines["naive-rag"], NaiveRAGPipeline)
    assert isinstance(lab.pipelines["advanced-rag"], HybridRAGPipeline)


def test_lab_preserves_a_valid_falsey_injected_expander() -> None:
    expander = FalseyQueryExpander({"identity": ("authentication",)})
    lab = KnowledgeAugmentationLab(
        [trusted_document()],
        scopes={"public"},
        query_expander=expander,
    )

    result = lab.run("advanced-rag", "identity failure", top_k=1)

    assert lab.query_expander is expander
    assert "identity failure authentication" in result.trace[0].detail


def test_lab_unknown_strategy_lists_supported_registry_keys() -> None:
    lab = KnowledgeAugmentationLab([trusted_document()], scopes={"public"})

    with pytest.raises(
        ValueError,
        match="unknown strategy 'missing'; choose from: advanced-rag, naive-rag",
    ):
        lab.run("missing", "question")


def test_naive_pipeline_satisfies_protocol_contract() -> None:
    lab = KnowledgeAugmentationLab([trusted_document()], scopes={"public"})

    result = accepts_pipeline(lab.pipelines["naive-rag"])

    assert result.strategy == "naive-rag"


def test_graph_table_memory_and_tool_pipeline_adapters_are_explicit() -> None:
    graph = KnowledgeGraph([("RAG", "uses", "retrieval")])
    graph_result = GraphRAGDemoPipeline(graph).run("RAG", top_k=1)

    table = TableStore([{"revenue": 10}, {"revenue": 20}])
    table_result = TableAugmentedPipeline(table, column="revenue", operation="sum").run("total revenue")

    memory = MemoryStore()
    memory.remember("user:a", "RAG retrieves evidence")
    memory_result = MemoryAugmentedPipeline(memory, scope="user:a").run("retrieves", top_k=1)

    tools = ToolRegistry()
    tools.register("lookup", lambda query: f"fresh:{query}")
    tool_result = ToolAugmentedPipeline(
        tools,
        tool_name="lookup",
        argument_builder=lambda query: {"query": query},
    ).run("status")

    assert graph_result.answer == "RAG --uses--> retrieval"
    assert table_result.answer == "sum(revenue) = 30"
    assert memory_result.answer == "RAG retrieves evidence"
    assert tool_result.answer == "fresh:status"
    assert all(
        isinstance(result, AugmentationResult) for result in (graph_result, table_result, memory_result, tool_result)
    )


def test_graph_pipeline_deduplicates_facts_reached_from_multiple_entities() -> None:
    graph = KnowledgeGraph([("A", "connects", "B"), ("B", "connects", "C")])

    result = GraphRAGDemoPipeline(graph).run("A B", top_k=3)

    assert result.answer == "A --connects--> B B --connects--> C"
    assert result.trace[0].detail == "selected 2 structured facts"
