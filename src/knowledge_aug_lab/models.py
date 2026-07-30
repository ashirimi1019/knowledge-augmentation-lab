"""Shared domain models used by every augmentation strategy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class FrozenMetadata(dict[str, Any]):
    """Recursively immutable, JSON/pickle/dataclass-compatible metadata."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, _memo: dict[int, Any]) -> FrozenMetadata:
        return self

    def __reduce__(self) -> tuple[type[FrozenMetadata], tuple[dict[str, Any]]]:
        return (type(self), (dict(self),))


def _freeze_value(value: Any) -> Any:
    """Recursively detach and freeze common metadata container types."""

    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, complex):
        return complex(value)
    if isinstance(value, bytes):
        return bytes(value)
    if value is None:
        return None
    if isinstance(value, Sequence):
        return tuple(_freeze_value(item) for item in value)
    raise TypeError(f"unsupported metadata value type: {type(value).__name__}")


def _freeze_mapping(metadata: Mapping[Any, Any]) -> FrozenMetadata:
    frozen: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        normalized_key = str(key)
        if normalized_key in frozen:
            raise ValueError(f"metadata keys collide after normalization: {normalized_key!r}")
        frozen[normalized_key] = _freeze_value(value)
    return FrozenMetadata(frozen)


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_mapping(metadata)


@dataclass(frozen=True, slots=True)
class Document:
    """A source document before chunking."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("document id cannot be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("document text cannot be empty")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("document metadata must be a mapping")

        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span with source provenance."""

    id: str
    document_id: str
    text: str
    start: int
    end: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("chunk id cannot be empty")
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("chunk document_id cannot be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("chunk text cannot be empty")
        if isinstance(self.start, bool) or not isinstance(self.start, int) or self.start < 0:
            raise ValueError("chunk start must be a non-negative integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int) or self.end <= self.start:
            raise ValueError("chunk end must be greater than start")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("chunk metadata must be a mapping")

        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "document_id", self.document_id.strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked chunk returned by a retriever."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, Chunk):
            raise TypeError("chunk must be a Chunk")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        try:
            score = float(self.score)
        except OverflowError as exc:
            raise ValueError("score must be finite") from exc
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if not isinstance(self.retriever, str) or not self.retriever.strip():
            raise ValueError("retriever cannot be empty")


@dataclass(frozen=True, slots=True)
class TraceStep:
    """An inspectable stage in an augmentation pipeline."""

    name: str
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("trace name cannot be empty")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("trace detail cannot be empty")


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    """A grounded response plus the evidence and decisions that produced it."""

    strategy: str
    answer: str
    citations: Sequence[str]
    evidence: Sequence[Chunk]
    trace: Sequence[TraceStep]

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or not self.strategy.strip():
            raise ValueError("strategy cannot be empty")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer cannot be empty")
        if isinstance(self.citations, (str, bytes)) or not isinstance(self.citations, Sequence):
            raise TypeError("citations must be a sequence")
        if isinstance(self.evidence, (str, bytes)) or not isinstance(self.evidence, Sequence):
            raise TypeError("evidence must be a sequence")
        if isinstance(self.trace, (str, bytes)) or not isinstance(self.trace, Sequence):
            raise TypeError("trace must be a sequence")

        citations = tuple(self.citations)
        evidence = tuple(self.evidence)
        trace = tuple(self.trace)

        if any(not isinstance(item, str) or not item.strip() for item in citations):
            raise ValueError("citations must be nonempty strings")
        if len(citations) != len(set(citations)):
            raise ValueError("citations must be unique")
        if any(not isinstance(item, Chunk) for item in evidence):
            raise TypeError("evidence must contain only Chunk values")
        if any(not isinstance(item, TraceStep) for item in trace):
            raise TypeError("trace must contain only TraceStep values")
        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "trace", trace)
