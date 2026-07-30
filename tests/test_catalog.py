from knowledge_aug_lab.catalog import get_catalog


def test_catalog_covers_major_augmentation_families_and_tag_ambiguity() -> None:
    catalog = get_catalog()
    slugs = {entry.slug for entry in catalog}

    assert {
        "naive-rag",
        "advanced-rag",
        "hybrid-rag",
        "graph-rag",
        "agentic-rag",
        "self-rag",
        "corrective-rag",
        "adaptive-rag",
        "cag",
        "kag",
        "memory-augmented-generation",
        "tool-augmented-generation",
        "table-augmented-generation",
        "long-context-generation",
    } <= slugs
    tag_entries = [entry for entry in catalog if "TAG" in entry.aliases]
    assert {entry.slug for entry in tag_entries} == {
        "tool-augmented-generation",
        "table-augmented-generation",
    }
    assert all(entry.mechanism and entry.tradeoff for entry in catalog)
