"""Dependency-free text processing for transparent experiments."""

from __future__ import annotations

import re

from knowledge_aug_lab.models import Chunk, Document

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Normalize text into searchable lowercase terms."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class RecursiveChunker:
    """Split on natural boundaries, falling back to a hard character window."""

    def __init__(self, chunk_size: int = 700, overlap: int = 100) -> None:
        if chunk_size < 40:
            raise ValueError("chunk_size must be at least 40 characters")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        document_ids = [document.id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document ids must be unique")

        chunks: list[Chunk] = []
        for document in documents:
            for index, (start, end) in enumerate(self._windows(document.text)):
                raw_text = document.text[start:end]
                text = raw_text.strip()
                if text:
                    adjusted_start = start + (len(raw_text) - len(raw_text.lstrip()))
                    adjusted_end = end - (len(raw_text) - len(raw_text.rstrip()))
                    chunks.append(
                        Chunk(
                            id=f"{document.id}#{index}",
                            document_id=document.id,
                            text=text,
                            start=adjusted_start,
                            end=adjusted_end,
                            metadata=dict(document.metadata),
                        )
                    )
        return chunks

    def _windows(self, text: str) -> list[tuple[int, int]]:
        if len(text) <= self.chunk_size:
            return [(0, len(text))]

        windows: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            hard_end = min(start + self.chunk_size, len(text))
            end = hard_end
            if hard_end < len(text):
                candidates = [
                    text.rfind("\n\n", start, hard_end),
                    text.rfind(". ", start, hard_end),
                    text.rfind(" ", start, hard_end),
                ]
                boundary = max(candidates)
                if boundary > start + self.chunk_size // 2:
                    end = boundary + (1 if text[boundary : boundary + 2] == ". " else 0)
            windows.append((start, end))
            if end == len(text):
                break
            next_start = max(0, end - self.overlap)
            if next_start <= start:
                next_start = end
            start = next_start
        return windows
