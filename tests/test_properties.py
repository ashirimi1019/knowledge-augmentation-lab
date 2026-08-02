from __future__ import annotations

import math
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from knowledge_aug_lab.augmentation import ContextCache, MemoryStore, TableStore, ToolRegistry, ToolSpec
from knowledge_aug_lab.evaluation import evaluate_retrieval, ndcg_at_k
from knowledge_aug_lab.generation import ExtractiveGenerator
from knowledge_aug_lab.knowledge import KnowledgeGraph
from knowledge_aug_lab.models import AugmentationResult, Chunk, Document, RetrievalResult
from knowledge_aug_lab.pipelines import KnowledgeAugmentationLab
from knowledge_aug_lab.query_expansion import QueryExpander
from knowledge_aug_lab.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever
from knowledge_aug_lab.security import filter_authorized_documents
from knowledge_aug_lab.text import RecursiveChunker

PROPERTY = settings(max_examples=30, deadline=None)
NONBLANK = st.text(alphabet=string.ascii_letters + string.digits + " ", min_size=1, max_size=40).filter(str.strip)
IDENTIFIER = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,15}", fullmatch=True)


def chunk(id_: str = "doc#0", document_id: str = "doc", text: str = "grounded evidence") -> Chunk:
    return Chunk(id_, document_id, text, 0, len(text))


class OnePassDocuments:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("documents were traversed more than once")
        yield from self.documents


class FailingDocuments:
    def __iter__(self):
        yield Document("doc", "text", {"scopes": ["public"], "trusted": True})
        raise TypeError("sentinel iterator failure")


@PROPERTY
@given(identifier=NONBLANK, text=NONBLANK)
def test_property_document_identifiers_are_trimmed(identifier: str, text: str) -> None:
    document = Document(f"  {identifier}  ", text)

    assert document.id == identifier.strip()


@PROPERTY
@given(start=st.integers(min_value=0, max_value=10_000), width=st.integers(min_value=1, max_value=10_000))
def test_property_chunk_accepts_valid_offsets(start: int, width: int) -> None:
    value = Chunk("doc#0", "doc", "evidence", start, start + width)

    assert value.start >= 0
    assert value.end > value.start


@PROPERTY
@given(start=st.integers(min_value=0, max_value=10_000), delta=st.integers(max_value=0))
def test_property_chunk_rejects_nonpositive_spans(start: int, delta: int) -> None:
    with pytest.raises(ValueError, match="chunk end must be greater than start"):
        Chunk("doc#0", "doc", "evidence", start, start + delta)


@PROPERTY
@given(score=st.floats(allow_nan=False, allow_infinity=False, width=64))
def test_property_retrieval_result_accepts_finite_scores(score: float) -> None:
    result = RetrievalResult(chunk(), score, 1, "property")

    assert math.isfinite(result.score)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, 10**400])
def test_adversarial_nonfinite_and_overflow_scores_fail_closed(score: float | int) -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        RetrievalResult(chunk(), score, 1, "property")


@PROPERTY
@given(values=st.lists(st.integers(), max_size=20))
def test_property_metadata_sequences_are_detached(values: list[int]) -> None:
    document = Document("doc", "text", {"values": values})
    snapshot = tuple(values)
    values.append(999)

    assert document.metadata["values"] == snapshot


def test_adversarial_json_metadata_scalar_branches_are_immutable() -> None:
    document = Document(
        "doc",
        "text",
        {
            "bool": True,
            "int": 1,
            "float": 1.5,
            "none": None,
        },
    )

    assert document.metadata == {"bool": True, "int": 1, "float": 1.5, "none": None}


@pytest.mark.parametrize("value", [{"a", "b"}, 1 + 2j, b"data"])
def test_adversarial_non_json_metadata_branches_fail_closed(value: object) -> None:
    with pytest.raises(TypeError, match="unsupported metadata value type"):
        Document("doc", "text", {"value": value})


def test_adversarial_metadata_keys_fail_closed() -> None:
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        Document("doc", "text", {1: "value"})  # type: ignore[dict-item]

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other: object) -> bool:
            return False

    with pytest.raises(ValueError, match="metadata keys collide after normalization"):
        Document("doc", "text", {"scope": "a", Key("scope"): "b"})


def test_adversarial_model_metadata_requires_mappings() -> None:
    with pytest.raises(TypeError, match="document metadata must be a mapping"):
        Document("doc", "text", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="chunk metadata must be a mapping"):
        Chunk("doc#0", "doc", "text", 0, 4, [])  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["citations", "evidence", "trace"])
def test_adversarial_augmentation_collections_reject_nonsequences(field: str) -> None:
    values: dict[str, object] = {
        "strategy": "rag",
        "answer": "answer",
        "citations": [],
        "evidence": [],
        "trace": [],
    }
    values[field] = object()

    with pytest.raises(TypeError, match=f"{field} must be a sequence"):
        AugmentationResult(**values)  # type: ignore[arg-type]


@PROPERTY
@given(requested=IDENTIFIER, other=IDENTIFIER)
def test_property_security_scopes_are_isolated(requested: str, other: str) -> None:
    documents = [
        Document("requested", "visible", {"scopes": [requested], "trusted": True}),
        Document("other", "hidden", {"scopes": [other], "trusted": True}),
    ]

    allowed = filter_authorized_documents(documents, {requested})

    expected = ["requested"] if requested != other else ["requested", "other"]
    assert [document.id for document in allowed] == expected


@pytest.mark.parametrize("scopes", [[], [""], ["   "], ["public", ""]])
def test_adversarial_blank_acl_entries_fail_closed(scopes: list[str]) -> None:
    document = Document("doc", "text", {"scopes": scopes, "trusted": True})

    assert filter_authorized_documents([document], {"public"}) == []


def test_adversarial_security_rejects_non_document_members() -> None:
    with pytest.raises(TypeError, match="documents must contain only Document values"):
        filter_authorized_documents(["not-a-document"], {"public"})  # type: ignore[list-item]

    with pytest.raises(TypeError, match="documents must be an iterable"):
        filter_authorized_documents(None, {"public"})  # type: ignore[arg-type]


def test_adversarial_security_snapshots_iterables_once() -> None:
    documents = OnePassDocuments([Document("doc", "text", {"scopes": ["public"], "trusted": True})])

    assert filter_authorized_documents(documents, {"public"}) == documents.documents
    assert documents.iterations == 1


def test_adversarial_security_preserves_iterator_failures() -> None:
    with pytest.raises(TypeError, match="sentinel iterator failure"):
        filter_authorized_documents(FailingDocuments(), {"public"})


@PROPERTY
@given(identifier=NONBLANK)
def test_property_duplicate_document_ids_are_rejected(identifier: str) -> None:
    documents = [Document(identifier, "first"), Document(f" {identifier} ", "second")]

    with pytest.raises(ValueError, match="document ids must be unique"):
        RecursiveChunker().split(documents)


@PROPERTY
@given(identifier=NONBLANK)
def test_property_duplicate_chunk_ids_are_rejected(identifier: str) -> None:
    chunks = [chunk(identifier, "a"), chunk(f" {identifier} ", "b")]

    with pytest.raises(ValueError, match="chunk ids must be unique"):
        BM25Retriever().fit(chunks)


@PROPERTY
@given(query=NONBLANK)
def test_property_empty_corpora_return_empty_rankings(query: str) -> None:
    assert BM25Retriever().fit([]).retrieve(query) == []
    assert TfidfRetriever().fit([]).retrieve(query) == []


@PROPERTY
@given(query=NONBLANK)
def test_property_bm25_ties_break_by_chunk_id(query: str) -> None:
    chunks = [chunk("z", "z", "shared"), chunk("a", "a", "shared")]

    results = BM25Retriever().fit(chunks).retrieve(query, top_k=2)

    assert [result.chunk.id for result in results] == ["a", "z"]


@PROPERTY
@given(query=NONBLANK)
def test_property_tfidf_ties_break_by_chunk_id(query: str) -> None:
    chunks = [chunk("z", "z", "shared"), chunk("a", "a", "shared")]

    results = TfidfRetriever().fit(chunks).retrieve(query, top_k=2)

    assert [result.chunk.id for result in results] == ["a", "z"]


@PROPERTY
@given(sentence=NONBLANK)
def test_property_generator_deduplicates_repeated_sentences(sentence: str) -> None:
    text = f"{sentence.strip()} evidence."

    answer, citations = ExtractiveGenerator().generate(
        "evidence",
        [chunk("a#0", "a", text), chunk("b#0", "b", text)],
        max_sentences=3,
    )

    assert answer.count(text) == 1
    assert citations == ["a"]


@PROPERTY
@given(unexpected=IDENTIFIER.filter(lambda value: value != "query"))
def test_property_tool_registry_rejects_unexpected_arguments(unexpected: str) -> None:
    registry = ToolRegistry()
    registry.register("lookup", lambda query: query)

    with pytest.raises(ValueError, match="unexpected tool arguments"):
        registry.call("lookup", **{unexpected: "payload"})


@PROPERTY
@given(scope_a=IDENTIFIER, scope_b=IDENTIFIER.filter(bool))
def test_property_memory_recall_never_crosses_scopes(scope_a: str, scope_b: str) -> None:
    memory = MemoryStore()
    memory.remember(scope_a, "alpha private fact")

    recalled = memory.recall(scope_b, "alpha")

    assert recalled == (["alpha private fact"] if scope_a == scope_b else [])


@PROPERTY
@given(
    ranked=st.lists(IDENTIFIER, min_size=1, max_size=15, unique=True),
    additional_relevant=st.sets(IDENTIFIER, max_size=10),
    k=st.integers(min_value=1, max_value=20),
)
def test_property_retrieval_metrics_are_bounded(ranked: list[str], additional_relevant: set[str], k: int) -> None:
    relevant = additional_relevant | {ranked[0]}
    metrics = evaluate_retrieval(ranked, relevant, k)
    ndcg = ndcg_at_k(ranked, relevant, k)

    assert 0.0 <= metrics.recall_at_k <= 1.0
    assert 0.0 <= metrics.precision_at_k <= 1.0
    assert 0.0 <= metrics.reciprocal_rank <= 1.0
    assert 0.0 <= metrics.hit_rate_at_k <= 1.0
    assert 0.0 <= ndcg <= 1.0
    assert metrics.reciprocal_rank > 0.0
    assert metrics.hit_rate_at_k == 1.0
    assert ndcg > 0.0


@PROPERTY
@given(values=st.lists(st.floats(min_value=-1e100, max_value=1e100, allow_nan=False), min_size=1, max_size=20))
def test_property_table_aggregations_remain_finite(values: list[float]) -> None:
    table = TableStore([{"value": value} for value in values])

    for operation in ("sum", "mean", "min", "max"):
        assert math.isfinite(table.aggregate("value", operation).value)


@pytest.mark.parametrize("invalid", [True, False, 40.0, "40", None])
def test_adversarial_chunk_size_requires_an_integer(invalid: object) -> None:
    with pytest.raises(ValueError, match="chunk_size must be an integer"):
        RecursiveChunker(chunk_size=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [True, False, 1.0, "1", None])
def test_adversarial_chunk_overlap_requires_an_integer(invalid: object) -> None:
    with pytest.raises(ValueError, match="overlap must be an integer"):
        RecursiveChunker(overlap=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BM25Retriever(k1=True),
        lambda: HybridRetriever([BM25Retriever()], rrf_k=True),
    ],
)
def test_adversarial_retrieval_booleans_are_not_numeric_parameters(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(("query", "top_k"), [("", 1), ("query", True), ("query", 0)])
def test_adversarial_pipeline_run_inputs_fail_closed(query: str, top_k: object) -> None:
    lab = KnowledgeAugmentationLab([], scopes=set())

    with pytest.raises(ValueError):
        lab.run("naive-rag", query, top_k=top_k)  # type: ignore[arg-type]


def test_adversarial_blank_pipeline_strategy_fails_closed() -> None:
    with pytest.raises(ValueError, match="strategy cannot be empty"):
        KnowledgeAugmentationLab([], scopes=set()).run(" ", "query")


def test_adversarial_query_expander_rejects_nonstring_keys() -> None:
    with pytest.raises(TypeError, match="expansion keys must be strings"):
        QueryExpander({1: ("value",)})  # type: ignore[dict-item]


def test_adversarial_hybrid_configuration_overflow_branches_fail_closed() -> None:
    with pytest.raises(TypeError, match="retrievers must implement retrieve"):
        HybridRetriever([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="rrf_k must fit finite"):
        HybridRetriever([BM25Retriever()], rrf_k=10**400)
    with pytest.raises(ValueError, match="combined weights must produce finite"):
        HybridRetriever([BM25Retriever(), TfidfRetriever()], weights=[1e308, 1e308], rrf_k=0)


def test_adversarial_generator_rejects_nonchunks_and_ignores_empty_split_tail() -> None:
    with pytest.raises(TypeError, match="chunks must be a list of Chunk values"):
        ExtractiveGenerator().generate("query", ["bad"])  # type: ignore[list-item]

    answer, citations = ExtractiveGenerator().generate("RAG", [chunk(text="RAG.   ")])
    assert answer == "RAG. [doc]"
    assert citations == ["doc"]


def test_adversarial_budget_and_registry_error_branches() -> None:
    document = Document("doc", "too long")
    with pytest.raises(ValueError, match="exceeds the CAG budget"):
        ContextCache([document], max_characters=1)

    registry = ToolRegistry()
    with pytest.raises(TypeError, match="spec must be a ToolSpec"):
        registry.register_spec(object())  # type: ignore[arg-type]
    registry.register("tool", lambda: "ok")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_spec(ToolSpec("tool", "duplicate", lambda: "no", frozenset()))
    with pytest.raises(KeyError, match="not allowlisted"):
        registry.call("missing")


def test_adversarial_graph_validation_and_cycle_branches() -> None:
    with pytest.raises(TypeError, match="triples must be a list"):
        KnowledgeGraph(("bad",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="every triple must contain"):
        KnowledgeGraph([("short", "edge")])  # type: ignore[list-item]

    graph = KnowledgeGraph([("A", "links", "B"), ("B", "links", "A")])
    report = graph.community_reports()[0]
    assert report.entities == ("A", "B")
    assert len(graph.neighborhood("A", max_hops=5)) == 2
