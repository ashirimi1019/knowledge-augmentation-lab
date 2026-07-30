"""Small, inspectable evaluation metrics for retrieval and grounding."""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_aug_lab.text import tokenize

_STOPWORDS = {"a", "an", "and", "are", "in", "is", "of", "the", "to", "their"}


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_k: float
    reciprocal_rank: float
    precision_at_k: float


def evaluate_retrieval(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
    k: int = 5,
) -> RetrievalMetrics:
    if k < 1:
        raise ValueError("k must be positive")
    unique_ranked_document_ids = list(dict.fromkeys(ranked_document_ids))
    top = unique_ranked_document_ids[:k]
    hits = sum(document_id in relevant_document_ids for document_id in top)
    recall = hits / len(relevant_document_ids) if relevant_document_ids else 0.0
    precision = hits / k
    first_relevant_rank = next(
        (
            rank
            for rank, document_id in enumerate(unique_ranked_document_ids, start=1)
            if document_id in relevant_document_ids
        ),
        None,
    )
    return RetrievalMetrics(
        recall_at_k=recall,
        reciprocal_rank=1 / first_relevant_rank if first_relevant_rank else 0.0,
        precision_at_k=precision,
    )


def lexical_groundedness(answer: str, context: str) -> float:
    """Fraction of answer content terms supported by supplied context."""

    answer_terms = {_stem(term) for term in tokenize(answer) if term not in _STOPWORDS}
    context_terms = {_stem(term) for term in tokenize(context) if term not in _STOPWORDS}
    return len(answer_terms & context_terms) / len(answer_terms) if answer_terms else 0.0


def _stem(term: str) -> str:
    if len(term) > 4 and term.endswith("es"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term
