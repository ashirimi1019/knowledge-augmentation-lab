"""Shared protocol and validation for augmentation pipelines."""

from __future__ import annotations

from typing import Protocol

from knowledge_aug_lab.models import AugmentationResult


class AugmentationPipeline(Protocol):
    """A query-to-result augmentation mechanism."""

    def run(self, query: str, top_k: int = 3) -> AugmentationResult: ...


def validate_run_inputs(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query cannot be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
