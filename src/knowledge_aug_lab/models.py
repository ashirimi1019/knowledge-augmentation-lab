"""Shared domain models used by every augmentation strategy."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, NoReturn, cast


class FrozenMetadata(Mapping[str, Any]):
    """Recursively immutable metadata with plain-dict deepcopy serialization."""

    __slots__ = ("_items",)
    _items: tuple[tuple[str, Any], ...]

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("metadata must be a mapping")
        frozen: dict[str, Any] = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings")
            normalized_key = str(key)
            if normalized_key in frozen:
                raise ValueError(f"metadata keys collide after normalization: {normalized_key!r}")
            frozen[normalized_key] = _freeze_value(value)
        object.__setattr__(self, "_items", tuple(frozen.items()))

    def __setattr__(self, _name: str, _value: Any) -> NoReturn:
        raise TypeError("metadata is immutable")

    def __getitem__(self, key: str) -> Any:
        for stored_key, value in self._items:
            if stored_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        other_mapping = cast(Mapping[object, object], other)
        return dict(self.items()) == dict(other_mapping.items())

    def _immutable(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise TypeError("metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable

    def setdefault(self, key: str, default: Any = None, /) -> Any:
        self._immutable(key, default)

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._immutable(*args, **kwargs)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return deepcopy(dict(self._items), memo)

    def __reduce__(self) -> tuple[type[FrozenMetadata], tuple[dict[str, Any]]]:
        return (type(self), (dict(self._items),))


def _freeze_value(value: Any) -> Any:
    """Recursively detach and freeze common metadata container types."""

    if isinstance(value, Mapping):
        return _freeze_mapping(cast(Mapping[Any, Any], value))
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("metadata value must be finite")
        return normalized
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview, range, complex, set, frozenset)):
        value_type = type(cast(object, value)).__name__
        raise TypeError(f"unsupported metadata value type: {value_type}")
    if type(value) is list:
        return tuple(_freeze_value(item) for item in cast(list[Any], value))
    if type(value) is tuple:
        return tuple(_freeze_value(item) for item in cast(tuple[Any, ...], value))
    raise TypeError(f"unsupported metadata value type: {type(value).__name__}")


def _freeze_mapping(metadata: Mapping[Any, Any]) -> FrozenMetadata:
    return FrozenMetadata(cast(Mapping[str, Any], metadata))


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_mapping(metadata)


def freeze_tool_value(value: Any, *, allow_frozen_metadata: bool = False) -> Any:
    """Detach and freeze one strict finite tool argument or output value."""

    if type(value) is FrozenMetadata:
        if allow_frozen_metadata:
            return value
        raise TypeError("unsupported tool value type: FrozenMetadata")
    if type(value) is dict:
        frozen: dict[str, Any] = {}
        for key, item in cast(dict[Any, Any], value).items():
            if type(key) is not str:
                raise TypeError("tool dictionary keys must be exact strings")
            frozen[key] = freeze_tool_value(item)
        return FrozenMetadata(frozen)
    if type(value) is list:
        return tuple(freeze_tool_value(item) for item in cast(list[Any], value))
    if type(value) is tuple:
        return tuple(freeze_tool_value(item) for item in cast(tuple[Any, ...], value))
    if type(value) is str:
        return value
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("tool float values must be finite")
        return value
    if value is None:
        return None
    raise TypeError(f"unsupported tool value type: {type(value).__name__}")


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(frozen=True, slots=True)
class Document:
    """A source document before chunking."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("document id cannot be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("document text cannot be empty")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("document metadata must be a mapping")

        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def __deepcopy__(self, memo: dict[int, Any]) -> Document:
        copied = type(self)(self.id, self.text, self.metadata)
        memo[id(self)] = copied
        return copied


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span with source provenance."""

    id: str
    document_id: str
    text: str
    start: int
    end: int
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

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

    def __deepcopy__(self, memo: dict[int, Any]) -> Chunk:
        copied = type(self)(self.id, self.document_id, self.text, self.start, self.end, self.metadata)
        memo[id(self)] = copied
        return copied


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
    attributes: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("trace name cannot be empty")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("trace detail cannot be empty")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("trace attributes must be a mapping")
        object.__setattr__(self, "attributes", _freeze_metadata(self.attributes))

    def __deepcopy__(self, memo: dict[int, Any]) -> TraceStep:
        copied = type(self)(self.name, self.detail, self.attributes)
        memo[id(self)] = copied
        return copied


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
        evidence_chunk_ids = [item.id for item in evidence]
        if len(evidence_chunk_ids) != len(set(evidence_chunk_ids)):
            raise ValueError("evidence chunk ids must be unique")
        evidence_document_ids = {item.document_id for item in evidence}
        unsupported_citations = sorted(set(citations) - evidence_document_ids)
        if unsupported_citations:
            raise ValueError(f"citations must reference evidence documents: {unsupported_citations}")
        if any(not isinstance(item, TraceStep) for item in trace):
            raise TypeError("trace must contain only TraceStep values")
        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "trace", trace)
