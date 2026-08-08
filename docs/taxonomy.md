# Knowledge augmentation taxonomy

Acronyms in this field are not a standards body. This guide separates broad engineering patterns from paper-specific methods and says exactly what this repository implements.

## Retrieval-centered methods

| Method | Canonical mechanism | Status in this lab | Primary source |
|---|---|---|---|
| **Naive RAG** | Retrieve top-k chunks and inject them before generation. Modern “retrieve-then-stuff” systems are not identical to the original end-to-end RAG model. | BM25, deterministic chunk IDs, citations, and traces. | [Lewis et al.](https://arxiv.org/abs/2005.11401) |
| **Advanced RAG** | Umbrella for pre-retrieval transformations and post-retrieval reranking, filtering, compression, and context construction. | Query expansion, hybrid candidate retrieval, lexical reranking/filtering. | [RAG survey](https://arxiv.org/abs/2312.10997) |
| **Modular RAG** | Replaceable components connected as a workflow rather than one fixed chain. | Typed `Retriever` protocol, shared models, and traces. | [RAG survey](https://arxiv.org/abs/2312.10997) |
| **Hybrid RAG** | Combine distinct retrieval representations, commonly sparse lexical plus dense semantic, with score or rank fusion. | BM25 + TF-IDF: two lexical representations fused by weighted RRF. This is adjacent to, not a substitute for, sparse+dense semantic hybrid retrieval. | [DPR](https://arxiv.org/abs/2004.04906), [fusion analysis](https://arxiv.org/abs/2210.11934) |
| **Query transformation** | Rewrite, expand, decompose, generate multiple views, or create a hypothetical document. | Transparent deterministic expansion; multi-query and HyDE are roadmap profiles. | [HyDE](https://arxiv.org/abs/2212.10496), [Query2doc](https://arxiv.org/abs/2303.07678) |
| **Reranking** | A slower second-stage model rescores candidates from a fast first-stage retriever. | Lexical overlap reranker for mechanics; cross-encoder backend is a roadmap profile. | [monoT5](https://arxiv.org/abs/2003.06713), [ColBERT](https://arxiv.org/abs/2004.12832) |
| **Multi-hop RAG** | Alternate subquestion generation, retrieval, evidence accumulation, and stopping. | Architecture documented; a bounded IRCoT-style loop is planned. | [IRCoT](https://arxiv.org/abs/2212.10509) |

## Adaptive and agentic methods

| Method | Important naming boundary | Mechanism | Lab position |
|---|---|---|---|
| **CRAG** | Paper-specific. Generic relevance filtering should be called `crag-inspired` or corrective RAG. | Grade evidence, refine weak knowledge, and optionally fall back to another source. | Not implemented. The advanced path has only a positive-score gate and lexical reranking—no CRAG grader, refinement, or fallback. [Paper](https://arxiv.org/abs/2401.15884) · [code](https://github.com/HuskyInSalt/CRAG) |
| **Self-RAG** | A trained method, not any prompt that asks an LLM to critique itself. | A specially trained model emits retrieval and reflection tokens for relevance, support, and utility. | Explained but not falsely “reproduced.” [Paper](https://arxiv.org/abs/2310.11511) · [code](https://github.com/AkariAsai/self-rag) |
| **Reflective RAG** | Provider-neutral engineering pattern, distinct from faithful Self-RAG. | Prompt/model evaluator critiques evidence or answers and may retry. | Roadmap profile with bounded retries and traces. |
| **Adaptive RAG** | Umbrella rather than one algorithm. | Route no-retrieval, single-hop, or iterative retrieval by query complexity. | Not implemented. The current registry dispatches an explicitly requested strategy. |
| **Agentic RAG** | Emerging umbrella, not a single canonical implementation. | A controller selects sources/tools, observes, replans, and terminates under limits. | Tool registry and traces are implemented; a state controller is planned. [Survey](https://arxiv.org/abs/2501.09136) · [ReAct](https://arxiv.org/abs/2210.03629) |

## Graph and structured knowledge

### GraphRAG

“GraphRAG” can mean generic graph retrieval or Microsoft’s specific pipeline. Microsoft GraphRAG extracts entities and relationships, performs community detection, produces hierarchical community reports, and supports local and global search.

This lab implements **teaching-scale GraphRAG mechanics**: deterministic graph construction inputs, connected communities, local multi-hop traversal, and inspectable community reports. It does not claim Leiden clustering or LLM-generated hierarchical summaries.

Sources: [GraphRAG paper](https://arxiv.org/abs/2404.16130) · [Microsoft documentation](https://microsoft.github.io/graphrag/)

### KAG

“KAG” is overloaded. Generic **knowledge-augmented generation** uses structured graphs, ontologies, rules, or databases. Capitalized OpenSPG KAG is a specific professional-domain framework combining schema alignment, vector/graph retrieval, and logical reasoning.

This lab calls its runnable primitive **mini-KAG / KG-RAG**: typed triples and transparent multi-hop paths. It does not claim to reproduce OpenSPG.

Sources: [KAG paper](https://arxiv.org/abs/2409.13731) · [OpenSPG KAG](https://github.com/OpenSPG/KAG)

### Table-Augmented Generation (TAG)

The TAG database paradigm is more than embedding CSV rows. A database performs relational operations while an LM performs semantic operators that ordinary SQL cannot express. The runnable `TableStore` here demonstrates validated equality filters, finite `Real` numeric aggregation, detached copies of caller-owned structured rows, and immutable row-index provenance—the deterministic database half of the architecture.

Sources: [TAG paper](https://arxiv.org/abs/2408.14717) · [TAG-Bench](https://github.com/TAG-Research/TAG-Bench)

## Context and cache augmentation

### Long-context generation

Place all or much of the corpus directly in the model context. Retrieval and persistent caching are optional. This is a baseline, not automatically RAG and not automatically CAG. Evaluate evidence position because long contexts suffer from “lost in the middle.”

Sources: [Lost in the Middle](https://arxiv.org/abs/2307.03172) · [RULER](https://arxiv.org/abs/2404.06654)

### Cache-Augmented Generation (CAG)

The CAG paper preloads a bounded knowledge corpus and reuses the model’s KV cache, avoiding query-time retrieval. That requires a backend exposing actual prefix/KV caching plus invalidation by corpus, model, tokenizer, and system-prompt revision.

`ContextCache` is explicitly a **mechanism simulator**: it enforces a corpus budget and records builds/hits, but it does not pretend a cached Python string is `past_key_values`.

Source: [CAG paper](https://arxiv.org/abs/2412.15605)

## Memory and tools

### Memory augmentation

Memory is dynamic and policy-governed, unlike a static document index. Distinguish working memory, summaries, episodic events, semantic facts, and structured state. Production memory needs provenance, tenant scope, TTL, correction, conflict handling, and deletion.

The lab implements process-local scoped append and lexical relevance-ranked recall. It has no persistence, TTL, provenance records, correction, conflict handling, encryption, or deletion. Source: [MemGPT](https://arxiv.org/abs/2310.08560)

### Tool-Augmented Generation

A model emits a typed request to an external capability and conditions later reasoning on the observation. Tool authority must come from the application, never from retrieved text. The lab allowlists tool and keyword names, validates callable compatibility, and snapshots arguments before invocation.

The recursive tool-value domain is deliberately narrower than general document or trace metadata. It accepts only exact built-in `dict` values with exact `str` keys, exact built-in `list` and `tuple`, exact `str`, `bool`, `int`, finite `float`, and `None`. It rejects arbitrary `Mapping` implementations, `defaultdict`, `UserDict`, mapping proxies, and custom mapping, list, tuple, string, or numeric subclasses. `FrozenMetadata` is accepted only as an exact type for internal/idempotent `ToolResult` argument reconstruction, not as a caller argument nested inside kwargs or as a tool output. Supported dictionaries and sequences are detached into recursively immutable `FrozenMetadata`/tuple snapshots; ordinary exact dictionaries are still passed to the callable itself, so a tool may mutate caller-owned data while the audit record retains its pre-call values. Output validation occurs after invocation and therefore cannot roll back tool side effects. The lab does not enforce semantic per-tool schemas, timeouts, sandboxes, authorization, or durable audit logs.

Sources: [ReAct](https://arxiv.org/abs/2210.03629) · [Toolformer](https://arxiv.org/abs/2302.04761)

> **TAG ambiguity:** both Table-Augmented Generation and Tool-Augmented Generation are called TAG. Always spell out the meaning.

## Multimodal augmentation

Multimodal RAG retrieves across text, images, audio, video, tables, or document layout. A serious implementation needs modality-specific ingestion, cross-modal embeddings or routing, provenance, and evaluation. It is represented in the catalog and roadmap, not fabricated as a text-only demo.
