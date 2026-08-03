"""Versioned, deterministic evaluation fixtures and reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import IO, Any, NoReturn, cast

from knowledge_aug_lab.evaluation import evaluate_retrieval, lexical_groundedness, ndcg_at_k
from knowledge_aug_lab.models import Document
from knowledge_aug_lab.pipelines import KnowledgeAugmentationLab

_FIXTURE_RESOURCE = "fixtures/evaluation-fixture-v1.json"
_BASELINE_RESOURCE = "fixtures/evaluation-baseline-v1.json"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    query: str
    relevant_document_ids: tuple[str, ...]
    k: int


@dataclass(frozen=True, slots=True)
class EvaluationFixture:
    schema_version: int
    name: str
    documents: tuple[Document, ...]
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    query: str
    relevant_document_ids: tuple[str, ...]
    retrieved_document_ids: tuple[str, ...]
    citations: tuple[str, ...]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    hit_rate_at_k: float
    ndcg_at_k: float
    lexical_groundedness: float
    trace_steps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    case_count: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float
    mean_hit_rate_at_k: float
    mean_ndcg_at_k: float
    mean_lexical_groundedness: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: int
    fixture_name: str
    strategy: str
    cases: tuple[CaseEvaluation, ...]
    summary: EvaluationSummary


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object name: {key!r}")
        result[key] = value
    return result


def _strict_json_load(handle: IO[str]) -> object:
    return cast(
        object,
        json.load(
            handle,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        ),
    )


def _read_json(path: str | Path | None, default_resource: str, label: str) -> object:
    try:
        if path is None:
            resource = files("knowledge_aug_lab").joinpath(default_resource)
            with resource.open("r", encoding="utf-8") as handle:
                return _strict_json_load(handle)
        if isinstance(path, bool) or not isinstance(path, (str, Path)):
            raise TypeError(f"{label} path must be a string, Path, or None")
        with Path(path).open("r", encoding="utf-8") as handle:
            return _strict_json_load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], mapping)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields must be exactly: {', '.join(sorted(expected))}")


def _evaluation_metadata(value: object, label: str) -> dict[str, object]:
    metadata = _mapping(value, label)
    if metadata.get("trusted") is not True:
        raise ValueError(f"{label} trusted must be exactly true")
    raw_scopes = metadata.get("scopes")
    if not isinstance(raw_scopes, list) or not raw_scopes:
        raise ValueError(f"{label} scopes must be a nonempty JSON array")
    scopes = tuple(_nonempty_string(scope, f"{label} scope") for scope in cast(list[object], raw_scopes))
    if len(scopes) != len(set(scopes)):
        raise ValueError(f"{label} scopes must be unique")
    if "public" not in scopes:
        raise ValueError(f"{label} scopes must include 'public'")
    return {**metadata, "trusted": True, "scopes": scopes}


def load_evaluation_fixture(path: str | Path | None = None) -> EvaluationFixture:
    """Load and validate a versioned evaluation corpus and query set."""

    payload = _mapping(_read_json(path, _FIXTURE_RESOURCE, "evaluation fixture"), "evaluation fixture")
    _exact_keys(payload, {"schema_version", "name", "documents", "cases"}, "evaluation fixture")
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("evaluation fixture schema_version must be 1")
    name = _nonempty_string(payload["name"], "evaluation fixture name")

    documents: list[Document] = []
    for index, raw_document in enumerate(_list(payload["documents"], "evaluation documents")):
        document = _mapping(raw_document, f"evaluation document {index}")
        _exact_keys(document, {"id", "text", "metadata"}, f"evaluation document {index}")
        metadata = _evaluation_metadata(document["metadata"], f"evaluation document {index} metadata")
        documents.append(
            Document(
                _nonempty_string(document["id"], f"evaluation document {index} id"),
                _nonempty_string(document["text"], f"evaluation document {index} text"),
                cast(Mapping[str, Any], metadata),
            )
        )
    if not documents:
        raise ValueError("evaluation fixture must contain at least one document")
    document_ids = [document.id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("evaluation document ids must be unique")
    known_document_ids = set(document_ids)

    cases: list[EvaluationCase] = []
    for index, raw_case in enumerate(_list(payload["cases"], "evaluation cases")):
        case = _mapping(raw_case, f"evaluation case {index}")
        _exact_keys(
            case,
            {"id", "query", "relevant_document_ids", "k"},
            f"evaluation case {index}",
        )
        relevant_values = _list(
            case["relevant_document_ids"],
            f"evaluation case {index} relevant_document_ids",
        )
        relevant_ids = tuple(
            _nonempty_string(value, f"evaluation case {index} relevant document id") for value in relevant_values
        )
        if not relevant_ids:
            raise ValueError("evaluation cases must contain at least one relevant document id")
        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError("evaluation relevant document ids must be unique")
        if not set(relevant_ids) <= known_document_ids:
            raise ValueError("relevant document ids must reference fixture documents")
        k = case["k"]
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("evaluation case k must be a positive integer")
        cases.append(
            EvaluationCase(
                id=_nonempty_string(case["id"], f"evaluation case {index} id"),
                query=_nonempty_string(case["query"], f"evaluation case {index} query"),
                relevant_document_ids=relevant_ids,
                k=k,
            )
        )
    if not cases:
        raise ValueError("evaluation fixture must contain at least one case")
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation case ids must be unique")

    return EvaluationFixture(
        schema_version=1,
        name=name,
        documents=tuple(documents),
        cases=tuple(cases),
    )


def _mean(values: list[float]) -> float:
    return round(math.fsum(values) / len(values), 6)


def run_evaluation_fixture(
    fixture: EvaluationFixture,
    strategy: str = "advanced-rag",
) -> EvaluationReport:
    """Run a fixture deterministically and return a timestamp-free report."""

    if not isinstance(fixture, EvaluationFixture):
        raise TypeError("fixture must be an EvaluationFixture")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy cannot be empty")

    lab = KnowledgeAugmentationLab(list(fixture.documents), scopes={"public"})
    case_reports: list[CaseEvaluation] = []
    for case in fixture.cases:
        result = lab.run(strategy, case.query, top_k=case.k)
        retrieved_ids = tuple(dict.fromkeys(chunk.document_id for chunk in result.evidence))
        relevant_ids = set(case.relevant_document_ids)
        metrics = evaluate_retrieval(list(retrieved_ids), relevant_ids, k=case.k)
        context = " ".join(chunk.text for chunk in result.evidence)
        case_reports.append(
            CaseEvaluation(
                case_id=case.id,
                query=case.query,
                relevant_document_ids=case.relevant_document_ids,
                retrieved_document_ids=retrieved_ids,
                citations=tuple(result.citations),
                recall_at_k=round(metrics.recall_at_k, 6),
                precision_at_k=round(metrics.precision_at_k, 6),
                reciprocal_rank=round(metrics.reciprocal_rank, 6),
                hit_rate_at_k=round(metrics.hit_rate_at_k, 6),
                ndcg_at_k=round(ndcg_at_k(list(retrieved_ids), relevant_ids, case.k), 6),
                lexical_groundedness=round(
                    lexical_groundedness(result.answer, context, citation_ids=result.citations),
                    6,
                ),
                trace_steps=tuple(step.name for step in result.trace),
            )
        )

    summary = EvaluationSummary(
        case_count=len(case_reports),
        mean_recall_at_k=_mean([case.recall_at_k for case in case_reports]),
        mean_precision_at_k=_mean([case.precision_at_k for case in case_reports]),
        mean_reciprocal_rank=_mean([case.reciprocal_rank for case in case_reports]),
        mean_hit_rate_at_k=_mean([case.hit_rate_at_k for case in case_reports]),
        mean_ndcg_at_k=_mean([case.ndcg_at_k for case in case_reports]),
        mean_lexical_groundedness=_mean([case.lexical_groundedness for case in case_reports]),
    )
    return EvaluationReport(
        schema_version=1,
        fixture_name=fixture.name,
        strategy=strategy,
        cases=tuple(case_reports),
        summary=summary,
    )


def _freeze_json_lists(value: object) -> object:
    if isinstance(value, list):
        items = cast(list[object], value)
        return tuple(_freeze_json_lists(item) for item in items)
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise ValueError("evaluation baseline must contain only string object keys")
        return {cast(str, key): _freeze_json_lists(item) for key, item in mapping.items()}
    return value


def load_evaluation_baseline(path: str | Path | None = None) -> dict[str, object]:
    """Load an exact expected report for deterministic regression checks."""

    payload = _mapping(_read_json(path, _BASELINE_RESOURCE, "evaluation baseline"), "evaluation baseline")
    return cast(dict[str, object], _freeze_json_lists(payload))


def evaluation_report_dict(report: EvaluationReport) -> dict[str, object]:
    """Convert a report to a JSON-compatible dictionary."""

    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    return cast(dict[str, object], json.loads(json.dumps(asdict(report))))


def _type_exact_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        actual_mapping = cast(dict[object, object], actual)
        expected_mapping = cast(dict[object, object], expected)
        if any(type(key) is not str for key in (*actual_mapping, *expected_mapping)):
            return False
        return actual_mapping.keys() == expected_mapping.keys() and all(
            _type_exact_equal(value, expected_mapping[key]) for key, value in actual_mapping.items()
        )
    if isinstance(actual, tuple):
        actual_items = cast(tuple[object, ...], actual)
        expected_items = cast(tuple[object, ...], expected)
        return len(actual_items) == len(expected_items) and all(
            _type_exact_equal(value, expected_value)
            for value, expected_value in zip(actual_items, expected_items, strict=True)
        )
    return actual == expected


def evaluation_report_matches_baseline(report: EvaluationReport, baseline: dict[str, object]) -> bool:
    """Compare a report and baseline with exact recursive JSON value types."""

    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    if not isinstance(baseline, dict):
        raise TypeError("baseline must be a dictionary")
    return _type_exact_equal(asdict(report), baseline)
