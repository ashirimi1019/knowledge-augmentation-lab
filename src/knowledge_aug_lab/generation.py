"""Grounded generation adapters.

The default generator is extractive and deterministic so every demo runs without
an API key. Applications can replace it with an LLM while retaining evidence and
trace contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge_aug_lab.models import Chunk
from knowledge_aug_lab.text import tokenize

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+")
_ABBREVIATIONS = ("e.g.", "i.e.", "mr.", "mrs.", "dr.", "vs.")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "how",
    "is",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
}


@dataclass(frozen=True, slots=True)
class SentenceCandidate:
    overlap: int
    chunk_rank: int
    sentence_rank: int
    text: str
    document_id: str


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
        candidate = text[start : boundary.start() + 1]
        if any(candidate.casefold().endswith(abbreviation) for abbreviation in _ABBREVIATIONS):
            continue
        sentences.append(candidate)
        start = boundary.end()
    sentences.append(text[start:])
    return sentences


class ExtractiveGenerator:
    """Compose a cited answer only from high-overlap evidence sentences."""

    def generate(self, query: str, chunks: list[Chunk], max_sentences: int = 3) -> tuple[str, list[str]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if not isinstance(chunks, list) or any(not isinstance(chunk, Chunk) for chunk in chunks):
            raise TypeError("chunks must be a list of Chunk values")
        if isinstance(max_sentences, bool) or not isinstance(max_sentences, int) or max_sentences < 1:
            raise ValueError("max_sentences must be a positive integer")

        query_terms = set(tokenize(query)) - _STOPWORDS
        candidates: list[SentenceCandidate] = []
        for chunk_rank, chunk in enumerate(chunks):
            for sentence_rank, sentence in enumerate(_split_sentences(chunk.text)):
                clean = sentence.strip()
                if not clean:
                    continue
                overlap = len(query_terms & set(tokenize(clean)))
                candidates.append(
                    SentenceCandidate(
                        overlap=overlap,
                        chunk_rank=chunk_rank,
                        sentence_rank=sentence_rank,
                        text=clean,
                        document_id=chunk.document_id,
                    )
                )
        candidates.sort(
            key=lambda item: (
                -item.overlap,
                item.chunk_rank,
                item.sentence_rank,
                item.document_id,
            )
        )

        seen_sentences: set[str] = set()
        selected: list[SentenceCandidate] = []
        for candidate in candidates:
            normalized = " ".join(candidate.text.casefold().split())
            if candidate.overlap <= 0 or normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            selected.append(candidate)
            if len(selected) == max_sentences:
                break
        if not selected:
            return "I do not have enough grounded evidence to answer that question.", []

        citations: list[str] = []
        rendered: list[str] = []
        for candidate in selected:
            if candidate.document_id not in citations:
                citations.append(candidate.document_id)
            rendered.append(f"{candidate.text} [{candidate.document_id}]")
        return " ".join(rendered), citations
