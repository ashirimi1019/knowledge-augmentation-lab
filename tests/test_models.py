import json
import math
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import Any

import pytest

from knowledge_aug_lab.models import AugmentationResult, Chunk, Document, RetrievalResult, TraceStep


def valid_chunk() -> Chunk:
    return Chunk("doc#0", "doc", "grounded evidence", 0, 17, {"source": "test"})


class PerPassSequence(Sequence[Any]):
    """Expose a different value on each traversal to probe TOCTOU validation."""

    def __init__(self, *values: Any) -> None:
        self._values = values
        self._passes = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Any:
        if index != 0:
            raise IndexError(index)
        return self._values[min(self._passes, len(self._values) - 1)]

    def __iter__(self) -> Iterator[Any]:
        value = self._values[min(self._passes, len(self._values) - 1)]
        self._passes += 1
        return iter((value,))


@pytest.mark.parametrize("document_id", ["", "   ", 1, None])
def test_document_rejects_invalid_ids(document_id: object) -> None:
    with pytest.raises(ValueError, match="document id cannot be empty"):
        Document(document_id, "text")  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "   ", 1, None])
def test_document_rejects_invalid_text(text: object) -> None:
    with pytest.raises(ValueError, match="document text cannot be empty"):
        Document("doc", text)  # type: ignore[arg-type]


def test_document_normalizes_id_and_freezes_metadata() -> None:
    source = {"scopes": ["public"], "nested": {"trusted": True}}
    document = Document(" doc ", "text", source)
    source["scopes"].append("private")
    source["nested"]["trusted"] = False

    assert document.id == "doc"
    assert document.metadata["scopes"] == ("public",)
    assert document.metadata["nested"]["trusted"] is True
    with pytest.raises(TypeError):
        document.metadata["nested"]["trusted"] = False  # type: ignore[index]


def test_document_metadata_supports_standard_serialization() -> None:
    document = Document("doc", "text", {"scopes": ["public"], "nested": {"trusted": True}})

    assert json.loads(json.dumps(document.metadata)) == {
        "scopes": ["public"],
        "nested": {"trusted": True},
    }
    assert asdict(document)["metadata"]["scopes"] == ("public",)
    assert pickle.loads(pickle.dumps(document)) == document


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", " ", "chunk id cannot be empty"),
        ("document_id", " ", "chunk document_id cannot be empty"),
        ("text", " ", "chunk text cannot be empty"),
        ("start", -1, "chunk start must be a non-negative integer"),
        ("start", True, "chunk start must be a non-negative integer"),
        ("end", 0, "chunk end must be greater than start"),
        ("end", True, "chunk end must be greater than start"),
    ],
)
def test_chunk_rejects_invalid_fields(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "id": "doc#0",
        "document_id": "doc",
        "text": "text",
        "start": 0,
        "end": 4,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        Chunk(**values)  # type: ignore[arg-type]


def test_chunk_normalizes_identifiers_and_freezes_metadata() -> None:
    source = {"trusted": True}
    chunk = Chunk(" doc#0 ", " doc ", "text", 0, 4, source)
    source["trusted"] = False

    assert chunk.id == "doc#0"
    assert chunk.document_id == "doc"
    assert chunk.metadata["trusted"] is True
    with pytest.raises(TypeError):
        chunk.metadata["trusted"] = False  # type: ignore[index]


@pytest.mark.parametrize("score", [math.inf, -math.inf, math.nan])
def test_retrieval_result_rejects_nonfinite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        RetrievalResult(valid_chunk(), score, 1, "bm25")


@pytest.mark.parametrize("rank", [0, -1, True])
def test_retrieval_result_rejects_invalid_rank(rank: object) -> None:
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        RetrievalResult(valid_chunk(), 1.0, rank, "bm25")  # type: ignore[arg-type]


def test_retrieval_result_validates_chunk_score_and_retriever() -> None:
    with pytest.raises(TypeError, match="chunk must be a Chunk"):
        RetrievalResult("chunk", 1.0, 1, "bm25")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="score must be numeric"):
        RetrievalResult(valid_chunk(), True, 1, "bm25")
    with pytest.raises(ValueError, match="retriever cannot be empty"):
        RetrievalResult(valid_chunk(), 1.0, 1, " ")
    with pytest.raises(ValueError, match="score must be finite"):
        RetrievalResult(valid_chunk(), 10**400, 1, "bm25")


@pytest.mark.parametrize(
    ("name", "detail", "message"),
    [("", "detail", "trace name cannot be empty"), ("name", " ", "trace detail cannot be empty")],
)
def test_trace_step_rejects_blank_fields(name: str, detail: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TraceStep(name, detail)


def test_augmentation_result_rejects_duplicate_or_blank_citations() -> None:
    chunk = valid_chunk()
    trace = TraceStep("retrieve", "selected evidence")
    with pytest.raises(ValueError, match="citations must be unique"):
        AugmentationResult("rag", "answer", ["doc", "doc"], [chunk], [trace])
    with pytest.raises(ValueError, match="citations must be nonempty strings"):
        AugmentationResult("rag", "answer", [" "], [chunk], [trace])
    with pytest.raises(ValueError, match="citations must be nonempty strings"):
        AugmentationResult("rag", "answer", [[]], [chunk], [trace])  # type: ignore[list-item]


def test_augmentation_result_validates_public_fields() -> None:
    chunk = valid_chunk()
    trace = TraceStep("retrieve", "selected evidence")
    with pytest.raises(ValueError, match="strategy cannot be empty"):
        AugmentationResult(" ", "answer", [], [chunk], [trace])
    with pytest.raises(ValueError, match="answer cannot be empty"):
        AugmentationResult("rag", " ", [], [chunk], [trace])
    with pytest.raises(TypeError, match="evidence must contain only Chunk values"):
        AugmentationResult("rag", "answer", [], ["chunk"], [trace])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="trace must contain only TraceStep values"):
        AugmentationResult("rag", "answer", [], [chunk], ["trace"])  # type: ignore[list-item]


def test_augmentation_result_snapshots_collections_as_immutable_sequences() -> None:
    chunk = valid_chunk()
    step = TraceStep("retrieve", "selected evidence")
    citations = ["doc"]
    evidence = [chunk]
    trace = [step]

    result = AugmentationResult("rag", "answer", citations, evidence, trace)
    citations.append("other")
    evidence.clear()
    trace.clear()

    assert result.citations == ("doc",)
    assert result.evidence == (chunk,)
    assert result.trace == (step,)
    with pytest.raises(AttributeError):
        result.citations.append("other")  # type: ignore[attr-defined]


def test_augmentation_result_snapshots_each_sequence_before_validation() -> None:
    chunk = valid_chunk()
    step = TraceStep("retrieve", "selected evidence")

    result = AugmentationResult(
        "rag",
        "answer",
        PerPassSequence("doc", "doc", []),
        PerPassSequence(chunk, "not-a-chunk"),
        PerPassSequence(step, "not-a-trace-step"),
    )

    assert result.citations == ("doc",)
    assert result.evidence == (chunk,)
    assert result.trace == (step,)
