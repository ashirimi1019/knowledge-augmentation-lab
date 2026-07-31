"""Sparse retrieval primitives implemented from first principles."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Protocol

from knowledge_aug_lab.models import Chunk, RetrievalResult
from knowledge_aug_lab.text import tokenize


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query cannot be empty")


def _validate_unique_chunk_ids(chunks: list[Chunk]) -> None:
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")
    if any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("chunks must contain only Chunk values")
    chunk_ids = [chunk.id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk ids must be unique")


class Retriever(Protocol):
    """Structural interface shared by all retrieval strategies."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...


class BM25Retriever:
    """Okapi BM25 retriever with no vector database or external service."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if isinstance(k1, bool) or not isinstance(k1, (int, float)):
            raise TypeError("k1 must be numeric")
        try:
            resolved_k1 = float(k1)
        except OverflowError as exc:
            raise ValueError("k1 must be finite and non-negative") from exc
        if not math.isfinite(resolved_k1) or k1 < 0:
            raise ValueError("k1 must be finite and non-negative")
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise TypeError("b must be numeric")
        try:
            resolved_b = float(b)
        except OverflowError as exc:
            raise ValueError("b must be finite and between 0 and 1") from exc
        if not math.isfinite(resolved_b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between 0 and 1")

        self.k1 = resolved_k1
        self.b = resolved_b
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
        _validate_query(query)
        _validate_top_k(top_k)
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
            normalization = 1 - self.b + self.b * length / max(self._average_length, 1.0)
            if self.k1 <= 1:
                saturation = frequency * (self.k1 + 1) / (frequency + self.k1 * normalization)
            else:
                saturation = frequency * (1 + 1 / self.k1) / (frequency / self.k1 + normalization)
            score += inverse_document_frequency * saturation
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
        _validate_query(query)
        _validate_top_k(top_k)
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
        if any(not callable(getattr(retriever, "retrieve", None)) for retriever in retrievers):
            raise TypeError("retrievers must implement retrieve")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
            raise ValueError("rrf_k must be a non-negative integer")
        try:
            resolved_rrf_k = float(rrf_k)
        except OverflowError as exc:
            raise ValueError("rrf_k must fit finite floating-point arithmetic") from exc
        if not math.isfinite(resolved_rrf_k):
            raise ValueError("rrf_k must fit finite floating-point arithmetic")

        resolved_weights = weights if weights is not None else [1.0] * len(retrievers)
        if len(resolved_weights) != len(retrievers):
            raise ValueError("weights must match retrievers")
        normalized_weights: list[float] = []
        for weight in resolved_weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError("weights must be finite positive numbers")
            try:
                normalized_weight = float(weight)
            except OverflowError as exc:
                raise ValueError("weights must be finite positive numbers") from exc
            if not math.isfinite(normalized_weight):
                raise ValueError("weights must be finite positive numbers")
            normalized_weights.append(normalized_weight)

        try:
            maximum_fused_score = math.fsum(weight / (resolved_rrf_k + 1.0) for weight in normalized_weights)
        except OverflowError as exc:
            raise ValueError("combined weights must produce finite fusion scores") from exc
        if not math.isfinite(maximum_fused_score):
            raise ValueError("combined weights must produce finite fusion scores")

        self.retrievers = list(retrievers)
        self.weights = normalized_weights
        self.rrf_k = rrf_k
        self._resolved_rrf_k = resolved_rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        _validate_query(query)
        _validate_top_k(top_k)
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        candidate_count = max(top_k * 3, top_k)
        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            for result in retriever.retrieve(query, candidate_count):
                if result.score <= 0:
                    continue
                existing = chunks.get(result.chunk.id)
                if existing is not None and existing != result.chunk:
                    raise ValueError(f"conflicting chunks share id {result.chunk.id!r}")
                scores[result.chunk.id] += weight / (self._resolved_rrf_k + result.rank)
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
