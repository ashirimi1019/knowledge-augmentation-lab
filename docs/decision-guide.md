# Decision guide

Start with the simplest mechanism that preserves the evidence and operations your task requires.

```text
Does the knowledge fit comfortably in context and change infrequently?
├─ yes → compare long-context against real KV/prefix-cache CAG
└─ no
   ├─ Is the answer primarily in unstructured documents?
   │  ├─ simple lookup → naive RAG baseline
   │  ├─ exact IDs + semantic language → hybrid RAG
   │  ├─ weak first-pass retrieval → transformations + reranking
   │  └─ multiple dependent facts → bounded multi-hop / agentic RAG
   ├─ Are explicit relationships, rules, or paths central?
   │  └─ KG-RAG / mini-KAG; GraphRAG for corpus-level themes
   ├─ Is the answer a relational calculation over rows?
   │  └─ validated SQL/table operations; TAG if semantic operators are needed
   ├─ Is fresh data, exact computation, or an action required?
   │  └─ tool augmentation with least privilege
   └─ Must user/project state persist across interactions?
      └─ scoped memory with provenance and deletion
```

## Trade-off matrix

| Mechanism | Corpus/data fit | Query-time latency | Freshness | Main risk |
|---|---|---:|---:|---|
| Naive RAG | medium/large documents | medium | high after reindex | bad retrieval becomes bad context |
| Advanced/hybrid RAG | heterogeneous documents | medium-high | high after reindex | pipeline tuning and extra latency |
| Long context | small/bounded corpus | high token cost | manual refresh | lost-in-the-middle, cost |
| True CAG | small, stable corpus | low when warm | cache invalidation | context/KV memory ceiling |
| KG-RAG / KAG | structured entity domains | medium-high | graph update dependent | entity resolution/schema cost |
| GraphRAG | relationship/global theme queries | high indexing cost | batch refresh | expensive extraction and summaries |
| Table/SQL | structured rows and metrics | low-medium | high | unsafe or wrong generated plans |
| Memory | evolving user/agent state | medium | immediate writes | stale/private/conflicting memory |
| Tools | fresh data or actions | variable | real-time | permissions, injection, side effects |

## Combine mechanisms deliberately

Common combinations: RAG + tools, RAG + graph, RAG + tables, RAG + memory, and CAG + tools. The combination should be visible in the trace. “Agentic” is not a substitute for bounded steps, permissions, stopping conditions, and evaluation.
