"""Hybrid sparse-retrieval RAG pipeline."""

from __future__ import annotations

from knowledge_aug_lab.generation import ExtractiveGenerator
from knowledge_aug_lab.models import AugmentationResult, TraceStep
from knowledge_aug_lab.query_expansion import QueryExpander
from knowledge_aug_lab.retrieval import HybridRetriever
from knowledge_aug_lab.text import tokenize

from .base import validate_run_inputs


class HybridRAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: ExtractiveGenerator,
        query_expander: QueryExpander,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.query_expander = query_expander

    def run(self, query: str, top_k: int = 3) -> AugmentationResult:
        validate_run_inputs(query, top_k)
        transformed = self.query_expander.expand(query)
        candidates = self.retriever.retrieve(transformed, top_k=top_k * 2)
        query_terms = set(tokenize(transformed))
        reranked = sorted(
            candidates,
            key=lambda result: (
                -len(query_terms & set(tokenize(result.chunk.text))),
                result.rank,
                result.chunk.id,
            ),
        )
        evidence = [result.chunk for result in reranked[:top_k] if result.score > 0]
        answer, citations = self.generator.generate(query, evidence)
        return AugmentationResult(
            strategy="advanced-rag",
            answer=answer,
            citations=citations,
            evidence=evidence,
            trace=[
                TraceStep(
                    "transform",
                    f"expanded query to: {transformed}",
                    {"original_query": query, "transformed_query": transformed},
                ),
                TraceStep(
                    "hybrid-retrieve",
                    "fused BM25 and TF-IDF rankings with RRF",
                    {"retrievers": ["bm25", "tfidf"], "candidate_count": len(candidates)},
                ),
                TraceStep(
                    "rerank-filter",
                    f"kept {len(evidence)} lexical-relevant candidates",
                    {
                        "requested_top_k": top_k,
                        "selected_chunk_ids": [chunk.id for chunk in evidence],
                    },
                ),
                TraceStep(
                    "generate",
                    "extractive generator composed a context-only answer",
                    {
                        "generator": "extractive",
                        "citation_ids": citations,
                        "abstained": not citations,
                    },
                ),
            ],
        )
