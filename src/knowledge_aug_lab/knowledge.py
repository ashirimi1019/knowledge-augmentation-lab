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
        self._outgoing: dict[str, list[Fact]] = defaultdict(list)
        self._names: dict[str, str] = {}
        self._facts: list[Fact] = []
        for subject, predicate, object_ in triples or []:
            self.add(subject, predicate, object_)

    def add(self, subject: str, predicate: str, object_: str) -> None:
        fact = Fact(subject.strip(), predicate.strip(), object_.strip())
        self._outgoing[subject.casefold()].append(fact)
        self._facts.append(fact)
        self._names[subject.casefold()] = subject.strip()
        self._names[object_.casefold()] = object_.strip()

    def entities(self) -> list[str]:
        return sorted(self._names.values(), key=str.casefold)

    def neighborhood(self, entity: str, max_hops: int = 2) -> list[Fact]:
        if max_hops < 1:
            return []
        queue: deque[tuple[str, int]] = deque([(entity.casefold(), 0)])
        visited = {entity.casefold()}
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
