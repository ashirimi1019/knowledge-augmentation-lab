"""Explicit, injectable lexical query expansion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from knowledge_aug_lab.text import tokenize

_DEFAULT_EXPANSIONS = {
    "authentication": ("login", "identity"),
    "credential": ("secret", "token"),
    "failure": ("error", "incident"),
    "rag": ("retrieval", "augmented", "generation"),
}


class QueryExpander:
    """Append configured lexical alternatives while preserving query order."""

    def __init__(self, expansions: Mapping[str, Sequence[str]] | None = None) -> None:
        source = _DEFAULT_EXPANSIONS if expansions is None else expansions
        if not isinstance(source, Mapping):
            raise TypeError("expansions must be a mapping")

        normalized: dict[str, tuple[str, ...]] = {}
        for raw_term, raw_expansions in dict(source).items():
            if not isinstance(raw_term, str):
                raise TypeError("expansion keys must be strings")
            term = raw_term.strip().casefold()
            if not term:
                raise ValueError("expansion keys cannot be empty")
            if term in normalized:
                raise ValueError(f"expansion keys collide after normalization: {term!r}")
            if isinstance(raw_expansions, (str, bytes)) or not isinstance(raw_expansions, Sequence):
                raise TypeError("expansion values must be sequences of strings")
            values = tuple(raw_expansions)
            if any(not isinstance(value, str) for value in values):
                raise TypeError("expansion values must contain only strings")
            stripped = tuple(value.strip().casefold() for value in values)
            if any(not value for value in stripped):
                raise ValueError("expansion values cannot be empty")
            normalized[term] = tuple(dict.fromkeys(stripped))
        self.expansions: Mapping[str, tuple[str, ...]] = MappingProxyType(normalized)

    def expand(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")

        additions: list[str] = []
        seen = set(tokenize(query))
        for term in tokenize(query):
            for expansion in self.expansions.get(term, ()):
                if expansion not in seen:
                    seen.add(expansion)
                    additions.append(expansion)
        return " ".join([query.strip(), *additions])
