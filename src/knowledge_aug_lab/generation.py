"""Grounded generation adapters.

The default generator is extractive and deterministic so every demo runs without
an API key. Applications can replace it with an LLM while retaining evidence and
trace contracts.
"""

from __future__ import annotations

import re

from knowledge_aug_lab.models import Chunk
from knowledge_aug_lab.text import tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
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


class ExtractiveGenerator:
    """Compose a cited answer only from high-overlap evidence sentences."""

    def generate(self, query: str, chunks: list[Chunk], max_sentences: int = 3) -> tuple[str, list[str]]:
        query_terms = set(tokenize(query)) - _STOPWORDS
        candidates: list[tuple[int, int, str, str]] = []
        for chunk_index, chunk in enumerate(chunks):
            for sentence in _SENTENCE_RE.split(chunk.text):
                clean = sentence.strip()
                if not clean:
                    continue
                overlap = len(query_terms & set(tokenize(clean)))
                candidates.append((overlap, -chunk_index, clean, chunk.document_id))
        candidates.sort(reverse=True)
        selected = [candidate for candidate in candidates if candidate[0] > 0][:max_sentences]
        if not selected:
            return "I do not have enough grounded evidence to answer that question.", []

        citations: list[str] = []
        rendered: list[str] = []
        for _, _, sentence, document_id in selected:
            if document_id not in citations:
                citations.append(document_id)
            rendered.append(f"{sentence} [{document_id}]")
        return " ".join(rendered), citations
