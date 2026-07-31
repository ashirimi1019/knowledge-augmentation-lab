"""Terminology catalog for the knowledge-augmentation landscape."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Concept:
    slug: str
    name: str
    family: str
    aliases: tuple[str, ...]
    mechanism: str
    best_for: str
    tradeoff: str
    implementation: str


def get_catalog() -> list[Concept]:
    """Return a curated taxonomy; acronyms are aliases, not universal standards."""

    return [
        Concept(
            "naive-rag",
            "Naive RAG",
            "retrieval",
            ("RAG",),
            "Chunk, index, retrieve top-k passages, then generate from injected context.",
            "Small document Q&A baselines and prototypes.",
            "Simple, but retrieval errors flow directly into generation.",
            "BM25 retrieval plus cited extractive generation.",
        ),
        Concept(
            "advanced-rag",
            "Advanced RAG",
            "retrieval",
            (),
            "Transform queries, combine retrievers, rerank, compress, and filter context.",
            "Production search where both recall and precision matter.",
            "More components, latency, and evaluation surface.",
            "Query expansion, BM25 + TF-IDF RRF, lexical reranking.",
        ),
        Concept(
            "modular-rag",
            "Modular RAG",
            "retrieval",
            (),
            "Connect replaceable retrievers, routers, rerankers, verifiers, memory, and generators.",
            "Systems that must evolve across data sources and model providers.",
            "Flexible graphs need typed contracts, traces, and configuration discipline.",
            "Retriever protocol, shared result models, and inspectable trace steps.",
        ),
        Concept(
            "hybrid-rag",
            "Hybrid RAG",
            "retrieval",
            (),
            "Fuse sparse exact-match and vector-space rankings.",
            "Mixed corpora with identifiers, jargon, and natural-language queries.",
            "Fusion weights and candidate depth require tuning.",
            "Weighted reciprocal-rank fusion.",
        ),
        Concept(
            "graph-rag",
            "GraphRAG",
            "retrieval",
            (),
            "Retrieve local subgraphs or precomputed community reports for global questions.",
            "Relationship and corpus-level questions.",
            "Graph extraction and community summarization are expensive.",
            "Graph traversal primitives and community-oriented documentation.",
        ),
        Concept(
            "multi-hop-rag",
            "Multi-hop RAG",
            "retrieval",
            (),
            "Alternate decomposition, retrieval, and evidence accumulation across bounded hops.",
            "Questions whose answer joins facts from multiple sources.",
            "Error propagation and runaway loops require stopping criteria.",
            "Architecture and trace contract; a production IRCoT loop is an extension profile.",
        ),
        Concept(
            "agentic-rag",
            "Agentic RAG",
            "retrieval",
            (),
            "A controller plans, selects sources, retrieves, judges sufficiency, and may retry.",
            "Multi-source and multi-hop workflows.",
            "Non-determinism, cost, loops, and observability burden.",
            "Not implemented; tool and trace primitives exist without an agent controller or source router.",
        ),
        Concept(
            "self-rag",
            "Self-RAG",
            "retrieval",
            (),
            "A specially trained model emits reflection decisions about retrieval and support.",
            "Mixed questions where retrieval is not always needed.",
            "True Self-RAG requires trained reflection tokens; prompting is only an analogue.",
            "Reflection checkpoints are documented without mislabeling heuristics as the paper model.",
        ),
        Concept(
            "corrective-rag",
            "Corrective RAG",
            "retrieval",
            ("CRAG",),
            "Grade retrieved evidence, refine weak queries, and use a fallback source.",
            "Hallucination-sensitive factual systems.",
            "Evaluator errors can reject useful context or trigger costly fallback.",
            "Not implemented; the advanced path has a positive-score gate, not CRAG grading or fallback retrieval.",
        ),
        Concept(
            "adaptive-rag",
            "Adaptive RAG",
            "retrieval",
            (),
            "Route no-retrieval, single-hop, or iterative retrieval by query complexity.",
            "Variable workloads with latency and quality budgets.",
            "Routing mistakes are an additional failure mode.",
            "Documented decision framework only; no adaptive controller is implemented.",
        ),
        Concept(
            "multimodal-rag",
            "Multimodal RAG",
            "retrieval",
            (),
            "Index and retrieve evidence across text, image, audio, video, or layouts.",
            "Reports, manuals, media archives, and visual QA.",
            "Cross-modal embeddings and modality-specific evaluation are harder.",
            "Architecture and evaluation extension points.",
        ),
        Concept(
            "cag",
            "Cache-Augmented Generation",
            "context",
            ("CAG",),
            "Preload a bounded stable corpus into long context or a reusable model KV cache.",
            "Repeated queries over a small, stable knowledge base.",
            "Context limits, cache invalidation, and lost-in-the-middle effects.",
            "A budgeted context-reuse simulator; true KV-cache reuse is an optional backend profile.",
        ),
        Concept(
            "long-context-generation",
            "Long-context generation",
            "context",
            (),
            "Place a large corpus directly in the prompt without a retrieval index.",
            "Small corpora and cross-document synthesis.",
            "Token cost and attention quality can degrade with length.",
            "Trade-offs are discussed in the decision guide; no empirical long-context comparison is implemented.",
        ),
        Concept(
            "kag",
            "Knowledge-Augmented Generation",
            "structured knowledge",
            ("KAG",),
            "Ground generation in graphs, ontologies, rules, or structured databases.",
            "Entity relations, constraints, and multi-hop reasoning.",
            "Schema construction, entity linking, and freshness are costly.",
            "Typed triples with deterministic multi-hop traversal.",
        ),
        Concept(
            "memory-augmented-generation",
            "Memory-Augmented Generation",
            "memory",
            ("MAG",),
            "Read and write scoped episodic or semantic memory across interactions.",
            "Personalized and long-running assistants.",
            "Privacy, stale memory, conflicts, and forgetting policies.",
            "Process-local scoped strings with lexical relevance-ranked recall; no persistence or deletion.",
        ),
        Concept(
            "tool-augmented-generation",
            "Tool-Augmented Generation",
            "action",
            ("TAG", "tool use"),
            "Select and call allowlisted external functions, APIs, search, or code execution.",
            "Real-time facts, exact calculations, and actions.",
            "Permissions, injection, side effects, and tool failures require controls.",
            "Allowlisted tool/keyword names and inspectable ToolResult; no value schema, timeout, or sandbox.",
        ),
        Concept(
            "table-augmented-generation",
            "Table-Augmented Generation",
            "structured knowledge",
            ("TAG", "table QA"),
            "Filter, aggregate, or reason over rows before verbalizing a result.",
            "Analytics, financial tables, and operational datasets.",
            "Schema ambiguity and generated-query validation.",
            "Validated equality filters and finite aggregations with row-index provenance.",
        ),
    ]
