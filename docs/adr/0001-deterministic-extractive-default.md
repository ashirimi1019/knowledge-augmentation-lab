# ADR 0001: Deterministic extractive default

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

A portfolio/reference lab must run in CI and for contributors without an API key while keeping answer grounding inspectable. A hosted or local generative model would introduce credentials, model drift, nondeterminism, cost, and hardware variance.

## Decision

The default generator selects source sentences deterministically, requires positive query overlap, attaches source IDs, deduplicates overlapping evidence, and abstains when no supported candidate exists. It is explicitly labeled an extractive baseline, not an LLM.

## Alternatives considered

- Hosted LLM by default: rejected because it makes core tests network/key dependent.
- Mock LLM output: rejected because fabricated model behavior teaches the wrong boundary.
- Local model dependency: deferred to an optional evaluated adapter due artifact size and platform variance.

## Consequences

- Grounding/citation mechanics and failures are reproducible.
- Fluency and semantic reasoning are intentionally limited.
- Production adapters must preserve evidence IDs, abstention, and trace contracts and add model/prompt revision plus latency/token/cost telemetry.

## Verification

`tests/test_generation_edge_cases.py`, `tests/test_pipelines.py`, and packaged `core-retrieval-v1` evaluation.
