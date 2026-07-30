"""Sparse retrieval primitives implemented from first principles."""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

from knowledge_aug_lab.models import Chunk, RetrievalResult
from knowledge_aug_lab.text import tokenize


def _validate_unique_chunk_ids(chunks: list[Chunk]) -> None:
    chunk_ids = [chunk.id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk ids must be unique")


class Retriever(Protocol):
    """Structural interface shared by all retrieval strategies."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...


class BM25Retriever:
    """Okapi BM25 retriever with no vector database or external service."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._average_length = 0.0

    def fit(self, chunks: list[Chunk]) -> BM25Retriever:
        _validate_unique_chunk_ids(chunks)
        self._chunks = list(chunks)
        self._term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._document_frequencies = Counter()
        for frequencies in self._term_frequencies:
            self._document_frequencies.update(frequencies.keys())
        self._average_length = (
            sum(sum(frequencies.values()) for frequencies in self._term_frequencies) / len(chunks) if chunks else 0.0
        )
        return self

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self._chunks:
            return []

        query_terms = tokenize(query)
        scored = [
            (self._score(query_terms, frequencies), chunk)
            for chunk, frequencies in zip(self._chunks, self._term_frequencies, strict=True)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [
            RetrievalResult(chunk=chunk, score=score, rank=rank, retriever="bm25")
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def _score(self, query_terms: list[str], frequencies: Counter[str]) -> float:
        length = sum(frequencies.values())
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            containing = self._document_frequencies[term]
            inverse_document_frequency = math.log(1 + (len(self._chunks) - containing + 0.5) / (containing + 0.5))
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / max(self._average_length, 1.0))
            score += inverse_document_frequency * frequency * (self.k1 + 1) / denominator
        return score


class TfidfRetriever:
    """Transparent cosine vector-space baseline for dense-retrieval experiments."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def fit(self, chunks: list[Chunk]) -> TfidfRetriever:
        _validate_unique_chunk_ids(chunks)
        self._chunks = list(chunks)
        document_frequencies: Counter[str] = Counter()
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        for terms in tokenized:
            document_frequencies.update(set(terms))
        corpus_size = 1 + len(chunks)
        self._idf = {term: math.log(corpus_size / (1 + count)) + 1 for term, count in document_frequencies.items()}
        self._vectors = [self._vector(terms) for terms in tokenized]
        return self

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self._vector(tokenize(query))
        scored = [
            (self._cosine(query_vector, vector), chunk)
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [
            RetrievalResult(chunk=chunk, score=score, rank=rank, retriever="tfidf")
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def _vector(self, terms: list[str]) -> dict[str, float]:
        counts = Counter(terms)
        return {term: count * self._idf.get(term, 0.0) for term, count in counts.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        dot = sum(value * right.get(term, 0.0) for term, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class HybridRetriever:
    """Fuse arbitrary ranked lists with weighted reciprocal-rank fusion (RRF)."""

    def __init__(
        self,
        retrievers: list[Retriever],
        weights: list[float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        if len(self.weights) != len(retrievers):
            raise ValueError("weights must match retrievers")
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        scores: Counter[str] = Counter()
        chunks: dict[str, Chunk] = {}
        candidate_count = max(top_k * 3, top_k)
        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            for result in retriever.retrieve(query, candidate_count):
                if result.score <= 0:
                    continue
                scores[result.chunk.id] += weight / (self.rrf_k + result.rank)
                chunks[result.chunk.id] = result.chunk
        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
        return [
            RetrievalResult(
                chunk=chunks[chunk_id],
                score=scores[chunk_id],
                rank=rank,
                retriever="hybrid-rrf",
            )
            for rank, chunk_id in enumerate(ranked, start=1)
        ]
