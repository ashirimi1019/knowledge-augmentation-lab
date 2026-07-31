# ADR 0002: Authorize before chunking and indexing

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

Filtering retrieved results after ranking allows unauthorized text to influence indexes, scores, caches, logs, and downstream context. A frozen model alone does not prevent nested ACL metadata mutation.

## Decision

`KnowledgeAugmentationLab` snapshots and validates documents, applies explicit trust and scope filtering, rejects duplicate authorized IDs, and only then chunks and fits retrievers. `Document` recursively detaches authorization metadata and normalizes accepted scalar/container subclasses.

## Alternatives considered

- Post-retrieval filtering: rejected because unauthorized content already influenced retrieval.
- Relying on a frozen dataclass: rejected because nested caller containers remain mutable.
- A universal security wrapper around all primitives: not claimed; standalone primitives retain explicit caller responsibilities.

## Consequences

- Unauthorized documents never reach the reference RAG indexes.
- Callers must provide trustworthy scope/trust metadata and use the protected orchestration path.
- Production systems still need an external identity/policy authority and source attestation.

## Verification

`tests/test_security.py`, authorization tests in `tests/test_pipelines.py`, metadata tests in `tests/test_models.py`, and security properties in `tests/test_properties.py`.
