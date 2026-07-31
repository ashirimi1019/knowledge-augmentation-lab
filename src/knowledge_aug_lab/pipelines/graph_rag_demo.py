"""Graph traversal teaching pipeline."""

from __future__ import annotations

from knowledge_aug_lab.knowledge import Fact, KnowledgeGraph
from knowledge_aug_lab.models import AugmentationResult, TraceStep

from .base import validate_run_inputs


class GraphRAGDemoPipeline:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def run(self, query: str, top_k: int = 3) -> AugmentationResult:
        validate_run_inputs(query, top_k)
        facts: list[Fact] = []
        seen_facts: set[tuple[str, str, str]] = set()
        for entity in self.graph.match_entities(query):
            for fact in self.graph.neighborhood(entity, max_hops=top_k):
                identity = (fact.subject.casefold(), fact.predicate.casefold(), fact.object.casefold())
                if identity not in seen_facts:
                    seen_facts.add(identity)
                    facts.append(fact)
            if len(facts) >= top_k:
                break
        selected = facts[:top_k]
        answer = " ".join(fact.render() for fact in selected) or "No matching graph relationships were found."
        return AugmentationResult(
            strategy="graph-rag",
            answer=answer,
            citations=[],
            evidence=[],
            trace=[TraceStep("graph-traverse", f"selected {len(selected)} structured facts")],
        )
