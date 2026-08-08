# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Private vulnerability-reporting policy and per-Python hash-pinned development/application constraints with separately pinned build tools.
- Adversarial regressions for malformed RRF rankings, forged citations, graph substring collisions, authorization metadata, mutable audit records, and expected CLI/app input errors.
- Canonical Git-tree source fingerprinting with verification mode and a machine-readable audit-attestation template.

### Changed

- Hybrid RRF now rejects duplicate chunk contributions and malformed source ranks; results require evidence-backed citations and unique evidence chunks.
- Graph entity lookup uses complete contiguous token phrases, and lexical groundedness excludes only recognized rendered citation markers.
- Table/tool snapshots are detached and immutable where publicly recorded; tool arguments and outputs use a dedicated exact-built-in recursive allowlist that rejects behavior-bearing mappings and subclasses, while document/trace metadata retains its separate normalization policy.
- Evaluation fixtures fail closed unless documents are trusted and public; the deterministic groundedness baseline is corrected from citation-format-distorted values to `1.0`.
- CI and release builds use hash-pinned per-interpreter environments and explicitly avoid unpinned PEP 517 build isolation.

## [0.2.0] - 2026-07-30

### Added

- Versioned deterministic evaluation fixture, exact baseline check, and `kal evaluate` command.
- Composition-based immutable, JSON-safe structured trace attributes for RAG stages.
- Property/adversarial tests with an enforced 95% branch-coverage floor.
- Strict Pyright, Python 3.10–3.13 CI, wheel/sdist installation checks, strict Twine validation, CodeQL, Dependabot, immutable Action pins, and protected required checks.
- Threat/control/test mapping, adversarial examples, ADRs, failure-mode mechanism guide, contributor gate, and semantic-tag release workflow.

### Changed

- Public models, retrievers, graph/table/cache/memory/tools, strict JSON evaluation fixtures, deterministic generation, and pipeline interfaces fail closed at their documented boundaries.
- Documentation distinguishes runnable primitives, teaching simulations, and unimplemented production mechanisms.

## [0.1.0] - 2026-07-30

### Added

- Initial dependency-light knowledge-augmentation lab, CLI, Streamlit app, GitHub Pages explorer, retrieval/evaluation primitives, showcase examples, tests, and CI.

[0.2.0]: https://github.com/ashirimi1019/knowledge-augmentation-lab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ashirimi1019/knowledge-augmentation-lab/releases/tag/v0.1.0
[Unreleased]: https://github.com/ashirimi1019/knowledge-augmentation-lab/compare/v0.2.0...HEAD
