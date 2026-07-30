"""Composable augmentation pipeline interfaces and teaching adapters."""

from .base import AugmentationPipeline
from .graph_rag_demo import GraphRAGDemoPipeline
from .hybrid_rag import HybridRAGPipeline
from .lab import KnowledgeAugmentationLab
from .memory_augmented import MemoryAugmentedPipeline
from .naive_rag import NaiveRAGPipeline
from .table_augmented import TableAugmentedPipeline
from .tool_augmented import ToolAugmentedPipeline

__all__ = [
    "AugmentationPipeline",
    "GraphRAGDemoPipeline",
    "HybridRAGPipeline",
    "KnowledgeAugmentationLab",
    "MemoryAugmentedPipeline",
    "NaiveRAGPipeline",
    "TableAugmentedPipeline",
    "ToolAugmentedPipeline",
]
