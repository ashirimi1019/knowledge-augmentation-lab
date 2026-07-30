# Evaluation and safety

## Evaluate each layer separately

| Layer | Questions | Metrics |
|---|---|---|
| Retrieval | Did the system find the supporting evidence? | Recall@k, Precision@k, MRR, nDCG |
| Context | Is context relevant, complete, non-redundant, and within budget? | context precision/recall, coverage, redundancy, tokens |
| Answer | Is the answer correct, supported, cited, and willing to abstain? | EM/F1, groundedness, citation precision/recall, abstention |
| Operations | Did tables/tools/graphs execute the intended plan? | plan accuracy, row/path provenance, tool success, retries |
| System | Is it fast and affordable enough? | p50/p95 latency, cost, tokens, memory, cache hit rate |

The included metrics are deterministic building blocks: Recall@k, Precision@k, MRR, and lexical groundedness. Production evaluation should combine labeled data, human review, and calibrated model judges rather than treating an LLM judge as ground truth.

Sources: [Ragas](https://arxiv.org/abs/2309.15217) · [ARES](https://arxiv.org/abs/2311.09476) · [BEIR](https://arxiv.org/abs/2104.08663)

## Minimal benchmark design

Stratify questions by lexical lookup, semantic paraphrase, multiple supporting passages, relationship/path reasoning, corpus themes, numerical/table aggregation, temporal conflicts, unanswerable questions, and adversarial/unauthorized evidence. Do not collapse methods that solve different task classes into one leaderboard score.

## Threat model

1. Poisoning or low-quality content enters the corpus.
2. Retrieved text performs indirect prompt injection.
3. Ranking before ACL filtering causes cross-tenant leakage.
4. An answer cites a source that does not support its claim.
5. Retrieved prose escalates tool privileges.
6. Incorrect/private facts persist in memory.

## Controls

- attach source, revision, trust, and scope metadata at ingestion;
- quarantine untrusted documents;
- apply ACLs **before** indexing/ranking;
- treat retrieved content as data, never as authority;
- validate citations against evidence;
- use allowlisted, least-privilege tools with typed schemas and timeouts;
- separate tool policy from retrieved prose;
- scope memory by tenant/user, record provenance, and support deletion;
- test abstention and adversarial cases in CI.

The runnable `filter_authorized_documents` primitive demonstrates the critical ordering rule: trust and access checks occur before retrieval.

Sources: [PoisonedRAG](https://arxiv.org/abs/2402.07867) · [OWASP Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) · [NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1)
