# Reproducible evaluation

The repository ships a deterministic regression suite named `core-retrieval-v1`. Run it with:

```bash
kal evaluate --check
```

The command loads the packaged fixture, runs the `advanced-rag` pipeline, computes a timestamp-free report, and compares it exactly with the packaged baseline. A mismatch exits with status 1. Custom fixtures and baselines are supported:

```bash
kal evaluate --fixture path/to/fixture.json --baseline path/to/baseline.json --check
```

## What this suite establishes

The fixture contains four public, trusted documents and three self-authored lexical queries. It establishes that, for this versioned corpus and configuration:

- ranking and generation are deterministic;
- the labeled document is retrieved in each case;
- metric conventions do not drift silently;
- trace stage contracts remain stable;
- the packaged wheel and source distribution contain working fixture data.

It is a **regression fixture, not a benchmark**. It does not demonstrate semantic generalization, external validity, production quality, model-judge reliability, adversarial robustness, latency, cost, multi-document support, table reasoning, temporal reasoning, or unanswerable-query behavior.

## Metric conventions

| Field | Convention |
|---|---|
| `recall_at_k` | Relevant unique documents returned in the first `k`, divided by all labeled relevant documents. |
| `precision_at_k` | Relevant unique documents returned, divided by the number actually returned up to `k`—not always by `k`. |
| `reciprocal_rank` | Reciprocal rank (RR) of the first relevant item in one case's full unique ranking. |
| `mean_reciprocal_rank` | Arithmetic mean of case-level RR values (MRR). |
| `hit_rate_at_k` | `1.0` when at least one relevant document appears in the first `k`, otherwise `0.0`. |
| `ndcg_at_k` | Binary-relevance nDCG over unique document IDs. |
| `lexical_groundedness` | Fraction of stemmed, non-stopword answer terms also present in selected context. It is not entailment. |

Duplicate ranked document IDs count only at their first occurrence. Metrics are rounded to six decimal places before baseline serialization. The report intentionally has no wall-clock timestamp, host data, or random seed because the implementation has no randomized stage.

The baseline's MRR and nDCG values of `1.0` do not imply perfect retrieval: each current case returns one extra document, so mean precision is `0.5`.

## Fixture schema

```json
{
  "schema_version": 1,
  "name": "example-v1",
  "documents": [
    {
      "id": "source-id",
      "text": "Nonempty source text.",
      "metadata": {"trusted": true, "scopes": ["public"]}
    }
  ],
  "cases": [
    {
      "id": "case-id",
      "query": "Nonempty question?",
      "relevant_document_ids": ["source-id"],
      "k": 1
    }
  ]
}
```

The strict loader rejects duplicate JSON object names, non-standard `NaN`/infinity constants, unknown/missing fields, non-integer or unsupported schema versions, malformed values, duplicate IDs, empty relevance sets, invalid `k`, and relevance labels that do not reference fixture documents.

## Updating a baseline

A changed report is a review event, not an automatic snapshot update:

1. Run `kal evaluate` and inspect every case-level difference.
2. Explain whether the change is an intended algorithm/fixture change or a regression.
3. Update the versioned fixture or baseline explicitly.
4. Run `kal evaluate --check` from source, wheel, and sdist via `python scripts/quality.py`.
5. Include the metric delta and rationale in the pull request.

For broader evaluation, add a separately named fixture with provenance, task strata, and held-out labels. Do not silently expand `core-retrieval-v1` and continue comparing unlike versions.
