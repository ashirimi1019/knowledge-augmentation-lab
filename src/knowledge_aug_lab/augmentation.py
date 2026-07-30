"""Non-retrieval augmentation primitives: cache, memory, tables, and tools."""

from __future__ import annotations

import inspect
import math
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
        if not isinstance(documents, list) or any(not isinstance(document, Document) for document in documents):
            raise TypeError("documents must be a list of Document values")
        if isinstance(max_characters, bool) or not isinstance(max_characters, int) or max_characters < 1:
            raise ValueError("max_characters must be a positive integer")
        rendered = "\n\n".join(f"[{document.id}] {document.text}" for document in documents)
        if len(rendered) > max_characters:
            raise ValueError(f"corpus has {len(rendered)} characters and exceeds the CAG budget of {max_characters}")
        self._context = rendered
        self.build_count = 1
        self.hit_count = 0

    def context_for(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
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
        if not isinstance(rows, list):
            raise TypeError("rows must be a list")
        if any(not isinstance(row, dict) for row in rows):
            raise TypeError("every row must be a dictionary")
        self.rows = [dict(row) for row in rows]

    def aggregate(
        self,
        column: str,
        operation: str,
        where: dict[str, Any] | None = None,
    ) -> TableResult:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("column cannot be empty")
        if operation not in {"count", "sum", "mean", "min", "max"}:
            raise ValueError(f"unsupported aggregation: {operation}")
        if where is not None and not isinstance(where, dict):
            raise TypeError("where must be a dictionary")

        filters = where or {}
        selected = [
            (index, row)
            for index, row in enumerate(self.rows)
            if all(row.get(key) == value for key, value in filters.items())
        ]
        if not selected:
            raise ValueError("no rows match the requested filters")
        provenance = [index for index, _ in selected]
        if any(column not in row for _, row in selected):
            raise ValueError(f"column {column!r} is not present in every selected row")

        if operation == "count":
            return TableResult(
                value=float(len(selected)),
                rows_used=len(selected),
                provenance=provenance,
            )

        values = [row.get(column) for _, row in selected]
        if not all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
            raise TypeError(f"column {column!r} must contain numeric values")

        numeric: list[float] = []
        for value in values:
            try:
                resolved_value = float(value)
            except OverflowError as exc:
                raise ValueError(f"column {column!r} must contain finite numeric values") from exc
            if not math.isfinite(resolved_value):
                raise ValueError(f"column {column!r} must contain finite numeric values")
            numeric.append(resolved_value)

        try:
            if operation == "sum":
                result = math.fsum(numeric)
            elif operation == "mean":
                scale = max(abs(value) for value in numeric)
                result = 0.0 if scale == 0 else scale * math.fsum(value / scale for value in numeric) / len(numeric)
            elif operation == "min":
                result = min(numeric)
            else:
                result = max(numeric)
        except OverflowError as exc:
            raise ValueError("aggregation result must be finite") from exc
        if not math.isfinite(result):
            raise ValueError("aggregation result must be finite")
        return TableResult(
            value=result,
            rows_used=len(selected),
            provenance=provenance,
        )


class MemoryStore:
    """User-scoped append-and-recall memory for MAG experiments."""

    def __init__(self) -> None:
        self._memories: dict[str, list[str]] = {}

    def remember(self, scope: str, fact: str) -> None:
        if not isinstance(scope, str) or not isinstance(fact, str) or not scope.strip() or not fact.strip():
            raise ValueError("scope and fact are required")
        clean_scope = scope.strip()
        clean_fact = fact.strip()
        memories = self._memories.setdefault(clean_scope, [])
        if clean_fact not in memories:
            memories.append(clean_fact)

    def recall(self, scope: str, query: str, top_k: int = 3) -> list[str]:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope cannot be empty")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = {self._stem(term) for term in tokenize(query)}
        candidates = self._memories.get(scope.strip(), [])
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


def _inspect_tool_signature(
    function: Callable[..., Any],
) -> tuple[frozenset[str], frozenset[str], frozenset[str], bool]:
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise ValueError("tool signature cannot be inspected") from exc

    keyword_arguments = frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    )
    required_keyword_arguments = frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        and parameter.default is inspect.Parameter.empty
    )
    required_positional_only = frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY and parameter.default is inspect.Parameter.empty
    )
    accepts_arbitrary_keywords = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    return keyword_arguments, required_keyword_arguments, required_positional_only, accepts_arbitrary_keywords


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A callable plus the exact keyword arguments callers may supply."""

    name: str
    description: str
    function: Callable[..., Any]
    allowed_arguments: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("tool names must be valid identifiers")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description cannot be empty")
        if not callable(self.function):
            raise TypeError("tool must be callable")
        if not isinstance(self.allowed_arguments, frozenset) or any(
            not isinstance(argument, str) or not argument.isidentifier() for argument in self.allowed_arguments
        ):
            raise ValueError("allowed arguments must be valid identifiers")
        accepted_arguments, required_arguments, required_positional_only, accepts_arbitrary_keywords = (
            _inspect_tool_signature(self.function)
        )
        if required_positional_only:
            names = ", ".join(sorted(required_positional_only))
            raise ValueError(f"tools cannot have required positional-only arguments: {names}")
        if not accepts_arbitrary_keywords and not self.allowed_arguments <= accepted_arguments:
            raise ValueError("allowed arguments must be accepted by the tool")
        missing_required = sorted(required_arguments - self.allowed_arguments)
        if missing_required:
            raise ValueError(f"required tool arguments must be allowlisted: {', '.join(missing_required)}")


class ToolRegistry:
    """Explicit allowlist for auditable tool-augmented generation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, function: Callable[..., Any]) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("tool names must be valid identifiers")
        if not callable(function):
            raise TypeError("tool must be callable")
        if name in self._tools:
            raise ValueError(f"tool is already registered: {name}")
        self._tools[name] = self._infer_spec(name, function)

    def register_spec(self, spec: ToolSpec) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("spec must be a ToolSpec")
        if spec.name in self._tools:
            raise ValueError(f"tool is already registered: {spec.name}")
        self._tools[spec.name] = spec

    def replace(self, name: str, function: Callable[..., Any]) -> None:
        if name not in self._tools:
            raise KeyError(f"tool is not registered: {name}")
        if not callable(function):
            raise TypeError("tool must be callable")
        self._tools[name] = self._infer_spec(name, function)

    def call(self, name: str, **arguments: Any) -> ToolResult:
        if name not in self._tools:
            raise KeyError(f"tool is not allowlisted: {name}")
        spec = self._tools[name]
        unexpected = sorted(set(arguments) - spec.allowed_arguments)
        if unexpected:
            raise ValueError(f"unexpected tool arguments: {', '.join(unexpected)}")
        output = spec.function(**arguments)
        return ToolResult(name=name, arguments=dict(arguments), output=output)

    @staticmethod
    def _infer_spec(name: str, function: Callable[..., Any]) -> ToolSpec:
        allowed_arguments, _, _, _ = _inspect_tool_signature(function)
        return ToolSpec(
            name=name,
            description=f"Registered tool {name}.",
            function=function,
            allowed_arguments=allowed_arguments,
        )
