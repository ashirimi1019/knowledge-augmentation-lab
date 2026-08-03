import json

import pytest

from knowledge_aug_lab.cli import main


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


def test_cli_demo_rejects_nonpositive_top_k_as_argument_error(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["demo", "question", "--top-k", "0"])

    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err
