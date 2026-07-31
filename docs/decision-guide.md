# Mechanism selection by failure mode

Choose the smallest mechanism that fixes an observed failure. The lab's default hybrid path remains lexical—BM25 plus TF-IDF with weighted RRF and lexical-overlap reranking—not dense semantic retrieval.

| Observable failure or requirement | Smallest next mechanism | Acceptance test | Lab support | Escalate when |
|---|---|---|---|---|
| Exact identifiers or distinctive terms are missed | BM25 tuning/chunk review | Relevant source enters top `k` for exact-ID cases | Implemented | Paraphrases still miss despite representative labels. |
| Two lexical representations rank useful evidence differently | BM25 + TF-IDF weighted RRF | Hybrid beats each lexical baseline on held-out cases | Implemented teaching baseline | Semantic paraphrases require embeddings/cross-encoders. |
| Retrieved candidates contain lexical noise | Deterministic overlap rerank/filter | Precision improves without unacceptable recall loss | Implemented | Relevance requires semantic judgment. |
| Query needs semantic paraphrase matching | Dense encoder plus evaluated vector index | Held-out semantic recall improves over BM25/TF-IDF | Roadmap | Domain shift or precision requires a cross-encoder. |
| Answer needs several dependent facts | Bounded multi-hop retrieval or graph traversal | Supporting paths are complete and hop-bounded | Graph primitive only | Entity resolution/schema quality dominates. |
| Answer is a relational calculation | Validated table/SQL plan | Result and source rows match oracle calculations | Equality filters and finite aggregations implemented | Joins, schemas, or semantic operators are required. |
| Stable bounded corpus repeats across requests | Long context, then real prefix/KV cache comparison | Warm latency/cost measured against retrieval baseline | `ContextCache` simulation only | Corpus exceeds context/cache budget or changes often. |
| Fresh data, exact computation, or an action is required | Least-privilege tool call | Typed/value-validated request, authorization, timeout, audit | Name/kwarg allowlist primitive only | Side effects require sandbox, approval, rollback. |
| User/project state must persist | Scoped memory | Recall is useful and never crosses scope; deletion works | In-memory scoped lexical recall only; no deletion | Persistence, consent, provenance, conflict handling are required. |
| Corpus-level themes are requested | Graph community pipeline | Reports are stable and answer labeled global questions | Connected-component simulation only | Faithful GraphRAG extraction/summarization is required. |
| Evidence is absent or unsupported | Abstain | Unanswerable set reaches target abstention precision/recall | Deterministic lexical abstention implemented | Semantic support or calibrated uncertainty is required. |

## Combination rules

- Combine mechanisms only when each addresses a measured failure.
- Keep authorization and provenance at the earliest shared boundary.
- Give every bounded step a stopping condition and explicit trace fields.
- Evaluate component metrics separately; do not hide retrieval, tool, or table failures inside one answer score.
- “Agentic,” “adaptive,” and “hybrid” are architectural claims that require implemented controllers or representations, not labels for a prompt loop.

## Current execution map

- `KnowledgeAugmentationLab` registers two integrated document pipelines: `naive-rag` and `advanced-rag`.
- Graph, table, memory, and tool adapters share the `AugmentationResult` contract but are constructed separately.
- `kal showcase` also invokes context-cache and other primitives directly.
- Self-RAG, CRAG, adaptive routing, dense retrieval, true model KV-cache CAG, faithful Microsoft GraphRAG, and OpenSPG KAG are roadmap items.
