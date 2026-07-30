from knowledge_aug_lab.knowledge import KnowledgeGraph


def test_kag_traverses_multi_hop_structured_knowledge() -> None:
    graph = KnowledgeGraph(
        [
            ("RAG", "uses", "Retrieval"),
            ("Retrieval", "supports", "Citations"),
            ("CAG", "avoids", "Per-query retrieval"),
        ]
    )

    facts = graph.neighborhood("RAG", max_hops=2)

    assert [fact.render() for fact in facts] == [
        "RAG --uses--> Retrieval",
        "Retrieval --supports--> Citations",
    ]


def test_graph_rag_builds_global_reports_from_connected_communities() -> None:
    graph = KnowledgeGraph(
        [
            ("RAG", "uses", "Retrieval"),
            ("Retrieval", "supports", "Citations"),
            ("CAG", "uses", "Context cache"),
        ]
    )

    reports = graph.community_reports()

    assert len(reports) == 2
    assert reports[0].entities == ("CAG", "Context cache")
    assert reports[1].entities == ("Citations", "RAG", "Retrieval")
    assert "RAG --uses--> Retrieval" in reports[1].summary
