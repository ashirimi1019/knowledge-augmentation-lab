"""Non-retrieval augmentation primitives: cache, memory, tables, and tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any

from knowledge_aug_lab.models import Document
from knowledge_aug_lab.text import tokenize


class ContextCache:
    """Preload a stable corpus once to demonstrate CAG context-reuse mechanics.

    This dependency-free class does not claim to expose a model's real KV cache.
    """

    def __init__(self, documents: list[Document], max_characters: int = 20_000) -> None:
        rendered = "\n\n".join(f"[{document.id}] {document.text}" for document in documents)
        if len(rendered) > max_characters:
            raise ValueError(f"corpus has {len(rendered)} characters and exceeds the CAG budget of {max_characters}")
        self._context = rendered
        self.build_count = 1
        self.hit_count = 0

    def context_for(self, query: str) -> str:
        if not query.strip():
            raise ValueError("query cannot be empty")
        self.hit_count += 1
        return self._context


@dataclass(frozen=True, slots=True)
class TableResult:
    value: float
    rows_used: int
    provenance: list[int]


class TableStore:
    """Typed filtering and aggregation with row-level provenance."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]

    def aggregate(
        self,
        column: str,
        operation: str,
        where: dict[str, Any] | None = None,
    ) -> TableResult:
        filters = where or {}
        selected = [
            (index, row)
            for index, row in enumerate(self.rows)
            if all(row.get(key) == value for key, value in filters.items())
        ]
        if not selected:
            raise ValueError("no rows match the requested filters")
        values = [row.get(column) for _, row in selected]
        if not all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
            raise TypeError(f"column {column!r} must contain numeric values")

        numeric = [float(value) for value in values]
        operations = {
            "count": lambda: float(len(numeric)),
            "sum": lambda: sum(numeric),
            "mean": lambda: sum(numeric) / len(numeric),
            "min": lambda: min(numeric),
            "max": lambda: max(numeric),
        }
        if operation not in operations:
            raise ValueError(f"unsupported aggregation: {operation}")
        return TableResult(
            value=operations[operation](),
            rows_used=len(selected),
            provenance=[index for index, _ in selected],
        )


class MemoryStore:
    """User-scoped append-and-recall memory for MAG experiments."""

    def __init__(self) -> None:
        self._memories: dict[str, list[str]] = {}

    def remember(self, scope: str, fact: str) -> None:
        if not scope.strip() or not fact.strip():
            raise ValueError("scope and fact are required")
        self._memories.setdefault(scope, []).append(fact.strip())

    def recall(self, scope: str, query: str, top_k: int = 3) -> list[str]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = {self._stem(term) for term in tokenize(query)}
        candidates = self._memories.get(scope, [])
        scored = (
            (
                len(query_terms & {self._stem(term) for term in tokenize(fact)}),
                index,
                fact,
            )
            for index, fact in enumerate(candidates)
        )
        ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1]))
        return [fact for _, _, fact in ranked[:top_k]]

    @staticmethod
    def _stem(term: str) -> str:
        return term[:-1] if len(term) > 3 and term.endswith("s") else term


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    arguments: dict[str, Any]
    output: Any


class ToolRegistry:
    """Explicit allowlist for auditable tool-augmented generation."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, function: Callable[..., Any]) -> None:
        if not name.isidentifier():
            raise ValueError("tool names must be valid identifiers")
        self._tools[name] = function

    def call(self, name: str, **arguments: Any) -> ToolResult:
        if name not in self._tools:
            raise KeyError(f"tool is not allowlisted: {name}")
        output = self._tools[name](**arguments)
        return ToolResult(name=name, arguments=dict(arguments), output=output)
