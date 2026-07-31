"""One-command execution of every implemented augmentation family."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from knowledge_aug_lab.augmentation import ContextCache, MemoryStore, TableStore, ToolRegistry
from knowledge_aug_lab.knowledge import KnowledgeGraph
from knowledge_aug_lab.models import AugmentationResult, Document
from knowledge_aug_lab.pipelines import KnowledgeAugmentationLab


def sample_documents() -> list[Document]:
    return [
        Document(
            "rag",
            "RAG retrieves query-relevant passages before generation and cites the selected evidence.",
            {"scopes": ["public"], "trusted": True},
        ),
        Document(
            "cag",
            "CAG preloads a bounded stable corpus into long context or a reusable key-value cache.",
            {"scopes": ["public"], "trusted": True},
        ),
        Document(
            "kag",
            "KAG uses structured knowledge such as graphs, ontologies, rules, and databases.",
            {"scopes": ["public"], "trusted": True},
        ),
    ]


def run_showcase() -> dict[str, dict[str, Any]]:
    """Execute local deterministic examples without an API key."""

    documents = sample_documents()
    lab = KnowledgeAugmentationLab(documents, scopes={"public"})
    question = "How does RAG ground generation?"
    naive = lab.run("naive-rag", question)
    advanced = lab.run("advanced-rag", question)

    cache = ContextCache(documents)
    cached_context = cache.context_for("Compare RAG and CAG")

    graph = KnowledgeGraph(
        [
            ("RAG", "uses", "Retrieval"),
            ("Retrieval", "returns", "Evidence"),
            ("Evidence", "supports", "Citations"),
            ("CAG", "uses", "Context cache"),
        ]
    )
    facts = graph.neighborhood("RAG", max_hops=3)
    reports = graph.community_reports()

    memory = MemoryStore()
    memory.remember("demo-user", "The user prefers local, inspectable AI systems.")
    memory.remember("demo-user", "The user is learning knowledge augmentation.")
    recalled = memory.recall("demo-user", "What AI system preference should be respected?", top_k=1)

    table = TableStore(
        [
            {"strategy": "RAG", "latency_ms": 120},
            {"strategy": "RAG", "latency_ms": 180},
            {"strategy": "CAG", "latency_ms": 35},
        ]
    )
    table_result = table.aggregate("latency_ms", "mean", {"strategy": "RAG"})

    tools = ToolRegistry()

    def estimate_cost(tokens: int, rate: float) -> float:
        return tokens / 1_000_000 * rate

    tools.register("estimate_cost", estimate_cost)
    tool_result = tools.call("estimate_cost", tokens=2_500_000, rate=0.40)

    def serialize_pipeline(result: AugmentationResult) -> dict[str, Any]:
        return {
            "answer": result.answer,
            "citations": result.citations,
            "evidence": [chunk.text for chunk in result.evidence],
            "trace": [asdict(step) for step in result.trace],
        }

    return {
        "naive-rag": serialize_pipeline(naive),
        "advanced-rag": serialize_pipeline(advanced),
        "cag": {
            "context_characters": len(cached_context),
            "cache_builds": cache.build_count,
            "cache_hits": cache.hit_count,
        },
        "kag": {"facts": [fact.render() for fact in facts]},
        "graph-rag": {
            "community_reports": [{"entities": report.entities, "summary": report.summary} for report in reports]
        },
        "memory-augmented": {"recalled": recalled},
        "table-augmented": asdict(table_result),
        "tool-augmented": asdict(tool_result),
    }
