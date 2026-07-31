from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from knowledge_aug_lab.cli import main
from knowledge_aug_lab.evaluation_suite import (
    evaluation_report_dict,
    evaluation_report_matches_baseline,
    load_evaluation_baseline,
    load_evaluation_fixture,
    run_evaluation_fixture,
)


def test_packaged_evaluation_fixture_reproduces_committed_baseline() -> None:
    fixture = load_evaluation_fixture()

    first = run_evaluation_fixture(fixture, strategy="advanced-rag")
    second = run_evaluation_fixture(fixture, strategy="advanced-rag")

    assert first == second
    assert first.fixture_name == "core-retrieval-v1"
    assert [case.case_id for case in first.cases] == ["rag-grounding", "cag-reuse", "graph-relations"]
    assert first.summary.case_count == 3
    assert asdict(first) == load_evaluation_baseline()


def test_evaluation_fixture_rejects_duplicate_document_ids(tmp_path) -> None:
    path = tmp_path / "duplicate-documents.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "invalid",
                "documents": [
                    {
                        "id": "duplicate",
                        "text": "First source.",
                        "metadata": {"trusted": True, "scopes": ["public"]},
                    },
                    {
                        "id": "duplicate",
                        "text": "Second source.",
                        "metadata": {"trusted": True, "scopes": ["public"]},
                    },
                ],
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation document ids must be unique"):
        load_evaluation_fixture(path)


def test_evaluation_fixture_rejects_unknown_relevant_document(tmp_path) -> None:
    path = tmp_path / "unknown-relevance.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "invalid",
                "documents": [
                    {
                        "id": "known",
                        "text": "Known source.",
                        "metadata": {"trusted": True, "scopes": ["public"]},
                    }
                ],
                "cases": [
                    {
                        "id": "case",
                        "query": "What is known?",
                        "relevant_document_ids": ["missing"],
                        "k": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relevant document ids must reference fixture documents"):
        load_evaluation_fixture(path)


def test_cli_evaluate_checks_and_emits_reproducible_baseline(capsys) -> None:
    exit_code = main(["evaluate", "--strategy", "advanced-rag", "--check"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["baseline_status"] == "passed"
    assert payload["fixture_name"] == "core-retrieval-v1"
    assert len(payload["cases"]) == 3


def _valid_fixture_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "validation-v1",
        "documents": [
            {
                "id": "source",
                "text": "A source contains evidence.",
                "metadata": {"trusted": True, "scopes": ["public"]},
            }
        ],
        "cases": [
            {
                "id": "case",
                "query": "What contains evidence?",
                "relevant_document_ids": ["source"],
                "k": 1,
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema_version=True), "schema_version must be 1"),
        (lambda payload: payload.update(schema_version=1.0), "schema_version must be 1"),
        (lambda payload: payload.update(name=" "), "fixture name must be a nonempty string"),
        (lambda payload: payload.update(documents=[]), "must contain at least one document"),
        (lambda payload: payload.update(documents={}), "evaluation documents must be a JSON array"),
        (lambda payload: payload.pop("name"), "fixture fields must be exactly"),
        (lambda payload: payload.update(cases=[]), "must contain at least one case"),
    ],
)
def test_evaluation_fixture_rejects_invalid_top_level_contract(tmp_path, mutate, message: str) -> None:
    payload = _valid_fixture_payload()
    mutate(payload)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_evaluation_fixture(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("relevant_document_ids", [], "at least one relevant document id"),
        ("relevant_document_ids", ["source", "source"], "relevant document ids must be unique"),
        ("k", True, "k must be a positive integer"),
        ("k", 0, "k must be a positive integer"),
    ],
)
def test_evaluation_fixture_rejects_invalid_case_contract(tmp_path, field: str, value: object, message: str) -> None:
    payload = _valid_fixture_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    case[field] = value
    path = tmp_path / "invalid-case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_evaluation_fixture(path)


def test_evaluation_fixture_rejects_duplicate_case_ids(tmp_path) -> None:
    payload = _valid_fixture_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    cases.append(dict(cases[0]))
    path = tmp_path / "duplicate-cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation case ids must be unique"):
        load_evaluation_fixture(path)


def test_evaluation_json_boundaries_reject_invalid_paths_and_payloads(tmp_path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")

    with pytest.raises(TypeError, match="path must be a string, Path, or None"):
        load_evaluation_fixture(True)
    with pytest.raises(ValueError, match="must contain valid JSON"):
        load_evaluation_fixture(invalid_json)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_evaluation_json_rejects_nonstandard_numeric_constants(tmp_path, constant: str) -> None:
    payload = _valid_fixture_payload()
    text = json.dumps(payload).replace('"schema_version": 1', f'"schema_version": {constant}')
    path = tmp_path / "nonstandard.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_evaluation_fixture(path)


def test_evaluation_json_rejects_duplicate_object_names(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object name"):
        load_evaluation_fixture(path)


def test_evaluation_runner_and_serializer_validate_public_types() -> None:
    with pytest.raises(TypeError, match="fixture must be an EvaluationFixture"):
        run_evaluation_fixture(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strategy cannot be empty"):
        run_evaluation_fixture(load_evaluation_fixture(), strategy=" ")
    with pytest.raises(TypeError, match="report must be an EvaluationReport"):
        evaluation_report_dict(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="report must be an EvaluationReport"):
        evaluation_report_matches_baseline(object(), {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="baseline must be a dictionary"):
        evaluation_report_matches_baseline(run_evaluation_fixture(load_evaluation_fixture()), [])  # type: ignore[arg-type]


def test_baseline_comparison_requires_exact_key_and_tuple_shapes() -> None:
    class StringSubclass(str):
        pass

    report = run_evaluation_fixture(load_evaluation_fixture())
    baseline = load_evaluation_baseline()
    assert evaluation_report_matches_baseline(report, baseline)

    subclass_key_baseline = dict(baseline)
    subclass_key_baseline[StringSubclass("schema_version")] = subclass_key_baseline.pop("schema_version")
    assert not evaluation_report_matches_baseline(report, subclass_key_baseline)

    cases = baseline["cases"]
    assert isinstance(cases, tuple)
    assert not evaluation_report_matches_baseline(report, {**baseline, "cases": cases[:-1]})
    assert not evaluation_report_matches_baseline(report, {**baseline, "cases": list(cases)})


def test_cli_evaluate_reports_exact_baseline_mismatch(tmp_path, capsys) -> None:
    baseline = tmp_path / "wrong-baseline.json"
    baseline.write_text("{}", encoding="utf-8")

    exit_code = main(["evaluate", "--check", "--baseline", str(baseline)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["baseline_status"] == "failed"


@pytest.mark.parametrize(
    ("location", "replacement"),
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("summary", "mean_recall_at_k"), True),
        (("summary", "mean_recall_at_k"), 1),
    ],
)
def test_cli_evaluate_rejects_type_confused_baselines(tmp_path, capsys, location, replacement) -> None:
    baseline = evaluation_report_dict(run_evaluation_fixture(load_evaluation_fixture()))
    target = baseline
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = replacement
    path = tmp_path / "type-confused-baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")

    exit_code = main(["evaluate", "--check", "--baseline", str(path)])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["baseline_status"] == "failed"


def test_cli_evaluate_can_emit_report_without_checking(capsys) -> None:
    assert main(["evaluate", "--strategy", "naive-rag"]) == 0
    assert json.loads(capsys.readouterr().out)["baseline_status"] == "not-checked"
