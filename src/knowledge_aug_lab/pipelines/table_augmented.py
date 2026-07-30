"""Typed table aggregation pipeline."""

from __future__ import annotations

from typing import Any

from knowledge_aug_lab.augmentation import TableStore
from knowledge_aug_lab.models import AugmentationResult, TraceStep

from .base import validate_run_inputs


class TableAugmentedPipeline:
    def __init__(
        self,
        table: TableStore,
        *,
        column: str,
        operation: str,
        where: dict[str, Any] | None = None,
    ) -> None:
        self.table = table
        self.column = column
        self.operation = operation
        self.where = dict(where) if where is not None else None

    def run(self, query: str, top_k: int = 3) -> AugmentationResult:
        validate_run_inputs(query, top_k)
        result = self.table.aggregate(self.column, self.operation, self.where)
        return AugmentationResult(
            strategy="table-augmented",
            answer=f"{self.operation}({self.column}) = {result.value:g}",
            citations=[],
            evidence=[],
            trace=[TraceStep("table-aggregate", f"used rows {result.provenance}")],
        )
