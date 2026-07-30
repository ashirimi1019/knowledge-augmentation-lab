import pytest

from knowledge_aug_lab.generation import ExtractiveGenerator, SentenceCandidate
from knowledge_aug_lab.models import Chunk


def chunk(id_: str, document_id: str, text: str) -> Chunk:
    return Chunk(id_, document_id, text, 0, len(text))


def test_sentence_candidate_exposes_named_ranking_semantics() -> None:
    candidate = SentenceCandidate(2, 0, 1, "Grounded sentence.", "rag")

    assert candidate.overlap == 2
    assert candidate.chunk_rank == 0
    assert candidate.sentence_rank == 1


@pytest.mark.parametrize("query", [None, "", "   "])
def test_generator_rejects_empty_queries(query: object) -> None:
    with pytest.raises(ValueError, match="query cannot be empty"):
        ExtractiveGenerator().generate(query, [])  # type: ignore[arg-type]


@pytest.mark.parametrize("max_sentences", [True, 0, -1, 1.5, None])
def test_generator_requires_positive_integer_sentence_limit(max_sentences: object) -> None:
    with pytest.raises(ValueError, match="max_sentences must be a positive integer"):
        ExtractiveGenerator().generate("grounding", [], max_sentences=max_sentences)  # type: ignore[arg-type]


def test_generator_abstains_for_empty_or_unsupported_evidence() -> None:
    generator = ExtractiveGenerator()
    expected = "I do not have enough grounded evidence to answer that question."

    assert generator.generate("How is RAG grounded?", []) == (expected, [])
    assert generator.generate("quokka zeppelin", [chunk("rag#0", "rag", "RAG retrieves evidence.")]) == (
        expected,
        [],
    )


def test_generator_deduplicates_repeated_sentences_across_overlapping_chunks() -> None:
    repeated = "RAG retrieves evidence before generation."
    answer, citations = ExtractiveGenerator().generate(
        "How does RAG retrieve evidence?",
        [chunk("rag#0", "rag", repeated), chunk("copy#0", "copy", repeated)],
        max_sentences=3,
    )

    assert answer.count(repeated) == 1
    assert citations == ["rag"]


def test_generator_ties_follow_chunk_then_sentence_order() -> None:
    answer, citations = ExtractiveGenerator().generate(
        "Which system retrieves evidence?",
        [
            chunk("first#0", "first", "Alpha retrieves evidence. Later retrieves evidence."),
            chunk("second#0", "second", "Beta retrieves evidence."),
        ],
        max_sentences=2,
    )

    assert answer.startswith("Alpha retrieves evidence. [first] Later retrieves evidence. [first]")
    assert citations == ["first"]


def test_abbreviations_remain_attached_to_their_sentence() -> None:
    answer, citations = ExtractiveGenerator().generate(
        "What passages does RAG use as evidence?",
        [chunk("rag#0", "rag", "RAG uses evidence, e.g. retrieved passages. It cites sources.")],
        max_sentences=1,
    )

    assert answer == "RAG uses evidence, e.g. retrieved passages. [rag]"
    assert citations == ["rag"]


def test_generator_preserves_nul_characters_in_source_evidence() -> None:
    answer, citations = ExtractiveGenerator().generate(
        "How does RAG retrieve evidence?",
        [chunk("rag#0", "rag", "RAG\x00retrieves evidence.")],
        max_sentences=1,
    )

    assert answer == "RAG\x00retrieves evidence. [rag]"
    assert citations == ["rag"]
