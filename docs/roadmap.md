# Roadmap: optional fidelity profiles

The default profile remains fast, transparent, and laptop-friendly. Heavyweight additions must be opt-in and accurately named.

## Retrieval
- sentence-transformer embeddings, persistent vector index, cross-encoder reranking;
- multi-query and HyDE ablations;
- BEIR subset with Recall@k and nDCG.

## Reasoning
- bounded IRCoT-style loop, supporting-fact evaluation, and stopping reasons;
- `crag-inspired` routing and a separate faithful CRAG checkpoint profile.

## Graph
- entity extraction/resolution, Leiden communities, hierarchical reports;
- Microsoft GraphRAG local/global comparison and OpenSPG KAG adapter.

## Structured data
- DuckDB read-only validated plans;
- Text2SQL vs row-RAG vs database TAG;
- bounded semantic operators and a TAG-Bench subset.

## Cache
- Hugging Face `past_key_values` or provider prefix caching;
- revision-aware invalidation;
- cold/warm latency, memory, and accuracy against RAG.

## Agent and multimodal
- typed state graph, budgets, loop detection, least-privilege tools, full trajectories;
- layout-aware PDFs, image/table/text routing, and modality-specific evaluation.
