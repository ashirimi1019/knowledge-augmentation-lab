# Threat model

## Scope

This threat model covers the dependency-light code in this repository. The primary protected path is `KnowledgeAugmentationLab`, which applies trust and scope filtering before chunking and fitting retrieval indexes. Standalone retrievers and graph, cache, table, memory, and tool primitives do not automatically inherit that authorization boundary.

The project has no model-connected agent controller, network tool executor, persistent database, secrets store, or multi-tenant identity provider. Controls below are narrow code-level demonstrations, not a production security certification.

## Assets and trust boundaries

**Assets**

- document text and authorization metadata;
- source/chunk identity and provenance;
- query and trace content;
- scoped memory strings;
- tool names, arguments, and outputs;
- deterministic evaluation and release artifacts.

**Trust boundaries**

1. Caller input → public models and fixture loaders.
2. Source metadata → authorization-before-indexing boundary.
3. Authorized documents → chunking and retrieval indexes.
4. Retrieved text → extractive generation and trace rendering.
5. Caller tool arguments → `ToolRegistry` allowlist/signature boundary.
6. Caller scope → `MemoryStore` partition.
7. Repository source → CI, package, and release automation.

## Threat, control, and regression mapping

| Threat | Demonstrated control | Regression evidence | Residual risk |
|---|---|---|---|
| **TM-01 Cross-scope document leakage** | **CTRL-01** trust/ACL filtering before RAG chunking and indexing | `test_acl_and_trust_filters_run_before_documents_reach_retrieval`; `test_pipeline_excludes_untrusted_and_unauthorized_documents_before_indexing`; `test_property_security_scopes_are_isolated` | Callers can bypass `KnowledgeAugmentationLab`; metadata is not an external authorization authority. |
| **TM-02 Malformed ACL bypass** | **CTRL-02** strict scope container/member validation; malformed entries fail closed | `test_acl_metadata_with_malformed_scope_shapes_fails_closed`; `test_adversarial_blank_acl_entries_fail_closed` | Trust/scopes remain caller-supplied assertions. |
| **TM-03 Metadata alias/TOCTOU** | **CTRL-03** composition-based recursive immutable snapshots, JSON-safe finite leaves, exact scalar normalization, one-pass sequence snapshots | `test_document_normalizes_id_and_freezes_metadata`; `test_metadata_cannot_be_mutated_through_dict_base_class_descriptors`; `test_metadata_rejects_non_json_leaf_values`; `test_metadata_snapshots_custom_sequences` | No signed source provenance or ingestion attestation. |
| **TM-04 Identity collision** | **CTRL-04** unique document/chunk IDs and conflicting hybrid provenance rejection | `test_duplicate_document_and_chunk_ids_are_rejected`; `test_pipeline_rejects_duplicate_authorized_document_ids`; property duplicate-ID tests | IDs are local, not globally namespaced. |
| **TM-05 Unsupported answer** | **CTRL-05** positive-score evidence gate and explicit extractive abstention | `test_generator_abstains_for_empty_or_unsupported_evidence`; `test_advanced_rag_abstains_when_query_has_no_matching_candidates` | Lexical overlap is not semantic support or entailment. |
| **TM-06 Tool argument escalation** | **CTRL-06** registered-name allowlist, exact keyword-name allowlist, callable signature validation, explicit replacement | `tests/test_tool_registry_security.py`; `test_property_tool_registry_rejects_unexpected_arguments` | No argument value schema, timeout, sandbox, caller authorization, output validation, or side-effect policy. |
| **TM-07 Cross-scope memory recall** | **CTRL-07** normalized exact-scope partitions and zero-overlap exclusion | `test_property_memory_recall_never_crosses_scopes`; memory tests in `test_augmentation.py` | No persistence, encryption, provenance, deletion, TTL, correction, or conflict handling. |
| **TM-08 Trace-content XSS** | **CTRL-08** HTML escaping in the Streamlit trace renderer | `test_trace_html_escapes_step_name_and_user_derived_detail` | Every other renderer must apply context-appropriate escaping. Traces may contain sensitive query text. |
| **TM-09 Numeric denial/corruption** | **CTRL-09** finite/range/type checks and overflow-safe arithmetic | `test_adversarial_nonfinite_and_overflow_scores_fail_closed`; retrieval/table/model boundary tests | Robustness control only; it does not detect adversarial document semantics. |
| **TM-10 Supply-chain drift** | **CTRL-10** immutable Action SHAs, strict CI, package installs, CodeQL, Dependabot, protected `main` | `tests/test_repository_policy.py`; live repository settings | Package dependencies use compatible ranges, not a fully locked reproducible environment. |

## Explicitly unsupported defenses

The repository does **not** claim to implement or test:

- trusted-but-poisoned content detection;
- indirect prompt-injection containment in an LLM;
- retrieval-to-tool privilege separation in a composed agent;
- secret-exfiltration prevention;
- semantic claim/citation entailment validation;
- tool timeout, sandboxing, rollback, or side-effect authorization;
- durable memory deletion, provenance, consent, or retention policy;
- adversarial coverage in `core-retrieval-v1`;
- cryptographic artifact signing or PyPI trusted publishing.

## Security review checklist

When extending a public boundary:

1. Define accepted types and semantic ranges.
2. Reject booleans as numbers where applicable.
3. Snapshot caller collections once before validation/storage.
4. Normalize or reject caller-controlled subclasses.
5. Authorize before derived data enters an index or cache.
6. Preserve source identity and reject collisions.
7. Add a red adversarial regression and sibling-path probes.
8. Verify serialization, strict typing, branch coverage, package installation, and trace rendering.
9. Update this matrix with the exact test symbol and residual risk.
