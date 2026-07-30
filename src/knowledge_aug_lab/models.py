"""Shared domain models used by every augmentation strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A source document before chunking."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("document id cannot be empty")
        if not self.text.strip():
            raise ValueError("document text cannot be empty")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span with source provenance."""

    id: str
    document_id: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked chunk returned by a retriever."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str


@dataclass(frozen=True, slots=True)
class TraceStep:
    """An inspectable stage in an augmentation pipeline."""

    name: str
    detail: str


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    """A grounded response plus the evidence and decisions that produced it."""

    strategy: str
    answer: str
    citations: list[str]
    evidence: list[Chunk]
    trace: list[TraceStep]
