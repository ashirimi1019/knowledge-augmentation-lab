"""Registry-based orchestration for document-backed RAG pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from knowledge_aug_lab.generation import ExtractiveGenerator
from knowledge_aug_lab.models import AugmentationResult, Document
from knowledge_aug_lab.query_expansion import QueryExpander
from knowledge_aug_lab.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever
from knowledge_aug_lab.security import filter_authorized_documents
from knowledge_aug_lab.text import RecursiveChunker

from .base import AugmentationPipeline
from .hybrid_rag import HybridRAGPipeline
from .naive_rag import NaiveRAGPipeline


class KnowledgeAugmentationLab:
    """Authorize once, construct composable pipelines, and dispatch by registry key."""

    def __init__(
        self,
        documents: list[Document],
        *,
        scopes: set[str],
        trusted_only: bool = True,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self.documents = filter_authorized_documents(documents, scopes, trusted_only)
        self.chunker = RecursiveChunker()
        self.chunks = self.chunker.split(self.documents)
        self.generator = ExtractiveGenerator()
        self.bm25 = BM25Retriever().fit(self.chunks)
        self.tfidf = TfidfRetriever().fit(self.chunks)
        self.hybrid = HybridRetriever([self.bm25, self.tfidf])
        self.query_expander = query_expander if query_expander is not None else QueryExpander()
        self._pipelines: dict[str, AugmentationPipeline] = {
            "naive-rag": NaiveRAGPipeline(
                self.documents,
                self.chunks,
                self.bm25,
                self.generator,
            ),
            "advanced-rag": HybridRAGPipeline(
                self.hybrid,
                self.generator,
                self.query_expander,
            ),
        }

    @property
    def pipelines(self) -> Mapping[str, AugmentationPipeline]:
        return MappingProxyType(self._pipelines)

    def run(self, strategy: str, query: str, top_k: int = 3) -> AugmentationResult:
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy cannot be empty")
        try:
            pipeline = self._pipelines[strategy]
        except KeyError as exc:
            supported = ", ".join(sorted(self._pipelines))
            raise ValueError(f"unknown strategy {strategy!r}; choose from: {supported}") from exc
        return pipeline.run(query, top_k)
