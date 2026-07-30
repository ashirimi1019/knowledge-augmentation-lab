"""Structured knowledge and graph-retrieval primitives."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fact:
    subject: str
    predicate: str
    object: str
    hop: int = 1

    def render(self) -> str:
        return f"{self.subject} --{self.predicate}--> {self.object}"


@dataclass(frozen=True, slots=True)
class CommunityReport:
    entities: tuple[str, ...]
    summary: str


class KnowledgeGraph:
    """Small explainable graph for local KAG and GraphRAG demonstrations."""

    def __init__(self, triples: list[tuple[str, str, str]] | None = None) -> None:
        if triples is not None and not isinstance(triples, list):
            raise TypeError("triples must be a list")
        self._outgoing: dict[str, list[Fact]] = defaultdict(list)
        self._names: dict[str, str] = {}
        self._facts: list[Fact] = []
        self._fact_keys: set[tuple[str, str, str]] = set()
        for triple in triples or []:
            if not isinstance(triple, tuple) or len(triple) != 3:
                raise ValueError("every triple must contain subject, predicate, and object")
            self.add(*triple)

    def add(self, subject: str, predicate: str, object_: str) -> None:
        values = {
            "subject": subject,
            "predicate": predicate,
            "object": object_,
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")

        clean_subject = subject.strip()
        clean_predicate = predicate.strip()
        clean_object = object_.strip()
        fact = Fact(clean_subject, clean_predicate, clean_object)
        subject_key = clean_subject.casefold()
        object_key = clean_object.casefold()
        fact_key = (subject_key, clean_predicate.casefold(), object_key)
        if fact_key in self._fact_keys:
            return

        self._outgoing[subject_key].append(fact)
        self._facts.append(fact)
        self._fact_keys.add(fact_key)
        self._names[subject_key] = clean_subject
        self._names[object_key] = clean_object

    def entities(self) -> list[str]:
        return sorted(self._names.values(), key=str.casefold)

    def neighborhood(self, entity: str, max_hops: int = 2) -> list[Fact]:
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("entity cannot be empty")
        if isinstance(max_hops, bool) or not isinstance(max_hops, int):
            raise TypeError("max_hops must be an integer")
        if max_hops < 1:
            return []
        entity_key = entity.strip().casefold()
        queue: deque[tuple[str, int]] = deque([(entity_key, 0)])
        visited = {entity_key}
        results: list[Fact] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for fact in self._outgoing.get(current, []):
                enriched = Fact(fact.subject, fact.predicate, fact.object, depth + 1)
                results.append(enriched)
                target = fact.object.casefold()
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))
        return results

    def match_entities(self, query: str) -> list[str]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        normalized = query.casefold()
        return [name for key, name in self._names.items() if key in normalized]

    def community_reports(self) -> list[CommunityReport]:
        """Create deterministic toy community reports from connected components.

        Production GraphRAG uses richer extraction, clustering, and LLM summaries;
        connected components keep those mechanics inspectable in this local lab.
        """

        adjacency: dict[str, set[str]] = defaultdict(set)
        for fact in self._facts:
            subject = fact.subject.casefold()
            object_ = fact.object.casefold()
            adjacency[subject].add(object_)
            adjacency[object_].add(subject)

        reports: list[CommunityReport] = []
        remaining = set(self._names)
        while remaining:
            seed = min(remaining)
            queue = deque([seed])
            component: set[str] = set()
            while queue:
                current = queue.popleft()
                if current in component:
                    continue
                component.add(current)
                queue.extend(sorted(adjacency[current] - component))
            remaining -= component
            entities = tuple(sorted((self._names[key] for key in component), key=str.casefold))
            facts = [
                fact.render()
                for fact in self._facts
                if fact.subject.casefold() in component and fact.object.casefold() in component
            ]
            reports.append(CommunityReport(entities=entities, summary="; ".join(facts)))
        return sorted(reports, key=lambda report: tuple(name.casefold() for name in report.entities))
