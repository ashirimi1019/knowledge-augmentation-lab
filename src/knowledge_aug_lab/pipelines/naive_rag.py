"""Naive BM25 RAG pipeline."""

from __future__ import annotations

from knowledge_aug_lab.generation import ExtractiveGenerator
from knowledge_aug_lab.models import AugmentationResult, Chunk, Document, TraceStep
from knowledge_aug_lab.retrieval import Retriever

from .base import validate_run_inputs


class NaiveRAGPipeline:
    def __init__(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        retriever: Retriever,
        generator: ExtractiveGenerator,
    ) -> None:
        self.documents = list(documents)
        self.chunks = list(chunks)
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 3) -> AugmentationResult:
        validate_run_inputs(query, top_k)
        results = self.retriever.retrieve(query, top_k=top_k)
        evidence = [result.chunk for result in results if result.score > 0]
        answer, citations = self.generator.generate(query, evidence)
        return AugmentationResult(
            strategy="naive-rag",
            answer=answer,
            citations=citations,
            evidence=evidence,
            trace=[
                TraceStep(
                    "chunk",
                    f"split {len(self.documents)} documents into {len(self.chunks)} chunks",
                    {"document_count": len(self.documents), "chunk_count": len(self.chunks)},
                ),
                TraceStep(
                    "retrieve",
                    f"BM25 selected {len(evidence)} query-relevant chunks",
                    {
                        "retriever": "bm25",
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
