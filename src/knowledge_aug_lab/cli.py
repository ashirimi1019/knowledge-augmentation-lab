"""Command-line entry point for the knowledge-augmentation lab."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from knowledge_aug_lab.catalog import get_catalog
from knowledge_aug_lab.evaluation_suite import (
    evaluation_report_dict,
    evaluation_report_matches_baseline,
    load_evaluation_baseline,
    load_evaluation_fixture,
    run_evaluation_fixture,
)
from knowledge_aug_lab.models import Document
from knowledge_aug_lab.pipelines import KnowledgeAugmentationLab
from knowledge_aug_lab.showcase import run_showcase

_SAMPLE_DOCUMENTS = [
    Document(
        "rag",
        "Retrieval-Augmented Generation retrieves query-relevant evidence before generation. "
        "Hybrid RAG combines sparse exact-match and vector-space retrieval with rank fusion.",
        {"scopes": ["public"], "trusted": True},
    ),
    Document(
        "cag",
        "Cache-Augmented Generation preloads a bounded stable corpus into long context or a "
        "reusable key-value cache, avoiding per-query retrieval.",
        {"scopes": ["public"], "trusted": True},
    ),
    Document(
        "kag",
        "Knowledge-Augmented Generation uses structured knowledge such as graphs, ontologies, "
        "rules, and databases for constrained or multi-hop reasoning.",
        {"scopes": ["public"], "trusted": True},
    ),
    Document(
        "tag",
        "TAG is overloaded. Tool-Augmented Generation calls typed external functions, while "
        "Table-Augmented Generation computes over structured rows with provenance.",
        {"scopes": ["public"], "trusted": True},
    ),
]


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kal",
        description="Explore retrieval, cache, graph, table, memory, and tool augmentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="Print the augmentation taxonomy")
    catalog.add_argument("--json", action="store_true", help="Emit JSON")

    subparsers.add_parser("showcase", help="Run every implemented family and emit JSON")

    demo = subparsers.add_parser("demo", help="Run a grounded RAG demo")
    demo.add_argument("query")
    demo.add_argument("--strategy", choices=["naive-rag", "advanced-rag"], default="advanced-rag")
    demo.add_argument("--top-k", type=_positive_integer, default=3)

    evaluate = subparsers.add_parser("evaluate", help="Run the versioned deterministic evaluation fixture")
    evaluate.add_argument("--strategy", choices=["naive-rag", "advanced-rag"], default="advanced-rag")
    evaluate.add_argument("--fixture", help="Use a custom evaluation fixture JSON file")
    evaluate.add_argument("--baseline", help="Use a custom exact baseline JSON file")
    evaluate.add_argument("--check", action="store_true", help="Fail when the report differs from the baseline")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "catalog":
        entries = [asdict(entry) for entry in get_catalog()]
        if args.json:
            print(json.dumps(entries, indent=2))
        else:
            for entry in entries:
                aliases = f" ({', '.join(entry['aliases'])})" if entry["aliases"] else ""
                print(f"{entry['name']}{aliases}: {entry['mechanism']}")
        return 0

    if args.command == "showcase":
        print(json.dumps(run_showcase(), indent=2))
        return 0

    if args.command == "evaluate":
        report = run_evaluation_fixture(
            load_evaluation_fixture(args.fixture),
            strategy=args.strategy,
        )
        payload = evaluation_report_dict(report)
        if args.check:
            matches = evaluation_report_matches_baseline(report, load_evaluation_baseline(args.baseline))
            payload["baseline_status"] = "passed" if matches else "failed"
            print(json.dumps(payload, indent=2))
            return 0 if matches else 1
        payload["baseline_status"] = "not-checked"
        print(json.dumps(payload, indent=2))
        return 0

    lab = KnowledgeAugmentationLab(_SAMPLE_DOCUMENTS, scopes={"public"})
    result = lab.run(args.strategy, args.query, top_k=args.top_k)
    payload = {
        "strategy": result.strategy,
        "answer": result.answer,
        "citations": result.citations,
        "evidence": [chunk.text for chunk in result.evidence],
        "trace": [asdict(step) for step in result.trace],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
