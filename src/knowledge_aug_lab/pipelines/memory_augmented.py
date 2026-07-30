"""Scoped in-memory recall pipeline."""

from __future__ import annotations

from knowledge_aug_lab.augmentation import MemoryStore
from knowledge_aug_lab.models import AugmentationResult, TraceStep

from .base import validate_run_inputs


class MemoryAugmentedPipeline:
    def __init__(self, memory: MemoryStore, *, scope: str) -> None:
        self.memory = memory
        self.scope = scope

    def run(self, query: str, top_k: int = 3) -> AugmentationResult:
        validate_run_inputs(query, top_k)
        recalled = self.memory.recall(self.scope, query, top_k=top_k)
        answer = " ".join(recalled) or "No relevant scoped memory was found."
        return AugmentationResult(
            strategy="memory-augmented",
            answer=answer,
            citations=[],
            evidence=[],
            trace=[TraceStep("memory-recall", f"recalled {len(recalled)} scoped facts")],
        )
