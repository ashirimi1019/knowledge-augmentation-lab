import json

import pytest

from knowledge_aug_lab.cli import build_parser, main


def test_cli_catalog_emits_machine_readable_taxonomy(capsys) -> None:
    exit_code = main(["catalog", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload) >= 14
    assert {item["slug"] for item in payload} >= {"naive-rag", "cag", "kag"}


def test_cli_showcase_runs_all_implemented_families(capsys) -> None:
    exit_code = main(["showcase"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert "graph-rag" in payload
    assert payload["cag"]["cache_hits"] == 1


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_cli_demo_rejects_blank_queries_as_argument_errors(query: str, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["demo", query])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage: kal demo" in captured.err
    assert "query must contain non-whitespace characters" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("value", ["0", "-1", "True", "False", "1.5", "not-an-integer"])
def test_cli_demo_rejects_invalid_top_k_as_argument_errors(value: str, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["demo", "question", "--top-k", value])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "positive integer" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "argv",
    [["demo"], ["demo", "question", "--strategy", "unsupported"]],
)
def test_cli_expected_parser_errors_do_not_print_tracebacks(argv: list[str], capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(argv)

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_demo_preserves_and_runs_valid_query(capsys) -> None:
    query = "  What is retrieval-augmented generation?  "
    parsed = build_parser().parse_args(["demo", query, "--top-k", "1"])

    assert parsed.query == query
    assert parsed.top_k == 1

    exit_code = main(["demo", query, "--top-k", "1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["strategy"] == "advanced-rag"
    assert len(payload["evidence"]) == 1
