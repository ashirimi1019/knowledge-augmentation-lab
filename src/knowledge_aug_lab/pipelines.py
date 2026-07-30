"""Composable, inspectable knowledge-augmentation pipelines."""

from __future__ import annotations

from knowledge_aug_lab.generation import ExtractiveGenerator
from knowledge_aug_lab.models import AugmentationResult, Document, TraceStep
from knowledge_aug_lab.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever
from knowledge_aug_lab.security import filter_authorized_documents
from knowledge_aug_lab.text import RecursiveChunker, tokenize


class KnowledgeAugmentationLab:
    """Run the same question through multiple augmentation strategies."""

    def __init__(
        self,
        documents: list[Document],
        *,
        scopes: set[str],
        trusted_only: bool = True,
    ) -> None:
        """Authorize source documents before any content is chunked or indexed."""

        self.documents = filter_authorized_documents(documents, scopes, trusted_only)
        self.chunker = RecursiveChunker()
        self.chunks = self.chunker.split(self.documents)
        self.generator = ExtractiveGenerator()
        self.bm25 = BM25Retriever().fit(self.chunks)
        self.tfidf = TfidfRetriever().fit(self.chunks)
        self.hybrid = HybridRetriever([self.bm25, self.tfidf])

    def run(self, strategy: str, query: str, top_k: int = 3) -> AugmentationResult:
        if strategy == "advanced-rag":
            return self._run_advanced(query, top_k)
        if strategy != "naive-rag":
            raise ValueError(f"unknown strategy: {strategy}")
        results = self.bm25.retrieve(query, top_k=top_k)
        evidence = [result.chunk for result in results if result.score > 0]
        answer, citations = self.generator.generate(query, evidence)
        return AugmentationResult(
            strategy=strategy,
            answer=answer,
            citations=citations,
            evidence=evidence,
            trace=[
                TraceStep(
                    "chunk",
                    f"split {len(self.documents)} documents into {len(self.chunks)} chunks",
                ),
                TraceStep("retrieve", f"BM25 selected {len(evidence)} query-relevant chunks"),
                TraceStep("generate", "extractive generator composed a context-only answer"),
            ],
        )

    def _run_advanced(self, query: str, top_k: int) -> AugmentationResult:
        transformed = self._expand_query(query)
        candidates = self.hybrid.retrieve(transformed, top_k=max(top_k * 2, top_k))
        query_terms = set(tokenize(transformed))
        reranked = sorted(
            candidates,
            key=lambda result: (
                -len(query_terms & set(tokenize(result.chunk.text))),
                result.rank,
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
                TraceStep("transform", f"expanded query to: {transformed}"),
                TraceStep("hybrid-retrieve", "fused BM25 and TF-IDF rankings with RRF"),
                TraceStep("rerank-filter", f"kept {len(evidence)} lexical-relevant candidates"),
                TraceStep("generate", "extractive generator composed a context-only answer"),
            ],
        )

    @staticmethod
    def _expand_query(query: str) -> str:
        expansions = {
            "authentication": "login identity",
            "credential": "secret token",
            "failure": "error incident",
            "rag": "retrieval augmented generation",
        }
        additions = [expansions[term] for term in tokenize(query) if term in expansions]
        return " ".join([query, *additions])
