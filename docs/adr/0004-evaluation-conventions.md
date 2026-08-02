# ADR 0004: Explicit evaluation conventions and versioned baselines

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

Retrieval metrics vary in duplicate handling and precision denominators. Unversioned snapshots can hide regressions, while timestamps/randomness make exact comparisons noisy.

## Decision

The lab deduplicates ranked document IDs by first occurrence; precision divides by results actually returned up to `k`; recall uses all labeled relevant documents; case output reports reciprocal rank; fixture summary reports mean reciprocal rank; nDCG uses binary relevance. The packaged report is rounded to six decimals, timestamp-free, and checked exactly against a versioned baseline.

## Alternatives considered

- Divide precision by `k` even when fewer results return: rejected for this educational contract, though common elsewhere.
- Tolerance-only snapshot comparison: rejected because the pipeline is deterministic.
- Present the small fixture as a benchmark: rejected due self-authored public-only lexical cases and no external validity.

## Consequences

- Metric changes become explicit reviewed artifacts.
- Comparisons with other systems require aligning conventions first.
- New task strata should use separately named/versioned fixtures.

## Verification

`tests/test_evaluation.py`, `tests/test_evaluation_conventions.py`, `tests/test_evaluation_suite.py`, and `kal evaluate --check`.
