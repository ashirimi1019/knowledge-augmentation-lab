# ADR 0003: Lexical hybrid and fidelity labels

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

“Hybrid RAG” often means sparse plus dense semantic retrieval, but the dependency-light lab implements BM25 and TF-IDF. Calling TF-IDF semantic or presenting teaching simulations as faithful paper reproductions would overstate the implementation.

## Decision

The `advanced-rag` path combines BM25 and TF-IDF rankings with weighted reciprocal-rank fusion, then applies lexical-overlap reranking. Documentation calls this a two-representation lexical hybrid baseline. Every broader mechanism receives an execution/fidelity label: faithful primitive, adjacent simulation, or roadmap/not implemented.

## Alternatives considered

- Add a mandatory embedding model/vector database: rejected for the default profile's dependency and artifact budget.
- Rename TF-IDF “semantic retrieval”: rejected as inaccurate.
- Remove broader concepts: rejected because mechanism comparison is an educational goal, provided fidelity is explicit.

## Consequences

- The core stays transparent and offline.
- Semantic paraphrase performance is a known limitation.
- Dense retrieval claims require a separately evaluated optional backend.

## Verification

`tests/test_hybrid.py`, `tests/test_retrieval.py`, `tests/test_pipeline_interface.py`, `docs/decision-guide.md`, and `docs/taxonomy.md`.
