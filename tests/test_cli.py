import json

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
