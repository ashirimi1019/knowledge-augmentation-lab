# Evaluation and safety

Evaluation and security are separate concerns with explicit scopes:

- [Reproducible evaluation](evaluation.md) defines implemented metrics, fixture schema, exact baseline behavior, and limitations.
- [Threat model](threat-model.md) maps demonstrated threats to controls, exact regression tests, and residual risks.
- [Adversarial examples](adversarial-examples.md) shows fail-closed inputs and non-defenses.

## Layered evaluation checklist

| Layer | Implemented here | Needed for broader claims |
|---|---|---|
| Retrieval | Recall@k, returned-result Precision@k, RR/MRR, binary nDCG@k | Held-out external datasets, semantic/multi-support strata |
| Context | Selected IDs and lexical groundedness input | Relevance, redundancy, coverage, token-budget evaluation |
| Answer | Deterministic extractive citations and abstention | Correctness/entailment, citation precision/recall, calibrated abstention |
| Operations | Row/path provenance and inspectable tool result primitives | Plan accuracy, retries, timeout, side-effect audits |
| System | Cache build/hit counters | p50/p95 latency, tokens, cost, memory, reliability |
| Security | Authorization ordering and adversarial boundary tests | Poisoning, model prompt injection, exfiltration, production IAM |

Do not collapse mechanisms that solve different task classes into one leaderboard score. Production evaluation should combine labeled data, human review, and calibrated model-assisted evaluation; an LLM judge is not ground truth.

Primary references: [Ragas](https://arxiv.org/abs/2309.15217), [ARES](https://arxiv.org/abs/2311.09476), [BEIR](https://arxiv.org/abs/2104.08663), [PoisonedRAG](https://arxiv.org/abs/2402.07867), [OWASP Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/), and [NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1).
