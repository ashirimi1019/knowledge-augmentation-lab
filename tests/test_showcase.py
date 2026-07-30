from knowledge_aug_lab.showcase import run_showcase


def test_showcase_executes_every_implemented_augmentation_family() -> None:
    output = run_showcase()

    assert set(output) == {
        "naive-rag",
        "advanced-rag",
        "cag",
        "kag",
        "graph-rag",
        "memory-augmented",
        "table-augmented",
        "tool-augmented",
    }
    assert output["naive-rag"]["citations"]
    assert output["cag"]["cache_hits"] == 1
    assert output["kag"]["facts"]
    assert output["graph-rag"]["community_reports"]
    assert output["table-augmented"]["rows_used"] == 2
    assert output["tool-augmented"]["output"] == 1.0
