"""Allowlisted tool invocation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from knowledge_aug_lab.augmentation import ToolRegistry
from knowledge_aug_lab.models import AugmentationResult, TraceStep

from .base import validate_run_inputs


class ToolAugmentedPipeline:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        tool_name: str,
        argument_builder: Callable[[str], dict[str, Any]],
    ) -> None:
        self.registry = registry
        self.tool_name = tool_name
        self.argument_builder = argument_builder

    def run(self, query: str, top_k: int = 3) -> AugmentationResult:
        validate_run_inputs(query, top_k)
        arguments = self.argument_builder(query)
        result = self.registry.call(self.tool_name, **arguments)
        return AugmentationResult(
            strategy="tool-augmented",
            answer=str(result.output),
            citations=[],
            evidence=[],
            trace=[TraceStep("tool-call", f"invoked allowlisted tool {self.tool_name}")],
        )
