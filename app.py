"""Streamlit visual explorer for the Knowledge Augmentation Lab."""

from __future__ import annotations

from dataclasses import asdict
from html import escape

import streamlit as st

from knowledge_aug_lab.catalog import get_catalog
from knowledge_aug_lab.evaluation import lexical_groundedness
from knowledge_aug_lab.evaluation_suite import evaluation_report_dict, load_evaluation_fixture, run_evaluation_fixture
from knowledge_aug_lab.pipelines import KnowledgeAugmentationLab
from knowledge_aug_lab.presentation import render_trace_step_html
from knowledge_aug_lab.showcase import run_showcase, sample_documents

st.set_page_config(
    page_title="Knowledge Augmentation Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root { --ink:#0d0d0d; --muted:#666; --green:#18e299; --line:rgba(0,0,0,.08); }
.stApp { background: #fff; color: var(--ink); }
[data-testid="stSidebar"] { background:#fafafa; border-right:1px solid var(--line); }
.block-container { max-width: 1240px; padding-top: 2.5rem; }
h1, h2, h3 { letter-spacing:-.035em; }
h1 { font-size:3.8rem !important; line-height:1.04 !important; font-weight:650 !important; }
h2 { margin-top:1.25rem !important; }
.hero-kicker { font:600 .73rem/1.4 ui-monospace, monospace; letter-spacing:.13em;
  text-transform:uppercase; color:#0b8059; background:#d4fae8; border-radius:999px;
  display:inline-block; padding:.35rem .72rem; }
.hero-copy { color:var(--muted); font-size:1.13rem; max-width:760px; line-height:1.65; }
.metric-card { border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem;
  background:#fff; box-shadow:0 2px 4px rgba(0,0,0,.03); min-height:116px; }
.metric-card b { font-size:1.9rem; display:block; letter-spacing:-.05em; }
.metric-card span { color:var(--muted); font-size:.84rem; }
.pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:.2rem .55rem;
  margin:.15rem .15rem .15rem 0; font:500 .7rem ui-monospace, monospace; }
.trace { border-left:3px solid var(--green); background:#fafafa; border-radius:0 12px 12px 0;
  padding:.72rem 1rem; margin:.45rem 0; }
.small-note { color:var(--muted); font-size:.84rem; }
div.stButton > button { border-radius:999px; border:1px solid var(--ink); background:var(--ink);
  color:#fff; padding:.45rem 1.15rem; }
div.stButton > button:hover { border-color:#0fa76e; background:#0fa76e; color:#fff; }
[data-testid="stMetric"] { border:1px solid var(--line); border-radius:16px; padding:1rem; }
</style>
""",
    unsafe_allow_html=True,
)

catalog = get_catalog()

with st.sidebar:
    st.markdown("### ◈ KAL")
    st.caption("Knowledge Augmentation Lab")
    page = st.radio("Explore", ["Concept map", "Live RAG lab", "All-family showcase", "Evaluation"])
    st.divider()
    family = st.selectbox(
        "Filter concept family",
        ["all", *sorted({concept.family for concept in catalog})],
        disabled=page != "Concept map",
    )
    st.caption("Laptop-first · API-key free · traced RAG paths")

st.markdown('<span class="hero-kicker">knowledge systems, made inspectable</span>', unsafe_allow_html=True)
st.title("Knowledge Augmentation Lab")
st.markdown(
    '<p class="hero-copy">Compare retrieval, reusable context, graphs, tables, memory, and tools '
    "without hiding the mechanics—or overstating what the acronyms mean.</p>",
    unsafe_allow_html=True,
)

if page == "Concept map":
    metrics = st.columns(4)
    cards = [
        (str(len(catalog)), "concepts mapped"),
        ("8", "showcase examples"),
        ("95%+", "branch coverage gate"),
        ("0", "required API keys"),
    ]
    for column, (value, label) in zip(metrics, cards, strict=True):
        column.markdown(
            f'<div class="metric-card"><b>{value}</b><span>{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("The complete landscape")
    selected = catalog if family == "all" else [c for c in catalog if c.family == family]
    for concept in selected:
        with st.container(border=True):
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"### {concept.name}")
                aliases = " ".join(f'<span class="pill">{escape(alias)}</span>' for alias in concept.aliases)
                if aliases:
                    st.markdown(aliases, unsafe_allow_html=True)
                st.write(concept.mechanism)
                st.caption(f"Best for: {concept.best_for}")
            with right:
                st.markdown("**Trade-off**")
                st.write(concept.tradeoff)
                st.markdown("**Lab implementation**")
                st.caption(concept.implementation)

elif page == "Live RAG lab":
    st.subheader("Run the same question through two retrieval architectures")
    question = st.text_input("Question", value="How does RAG ground generation?")
    strategy = st.radio(
        "Pipeline",
        ["naive-rag", "advanced-rag"],
        index=1,
        horizontal=True,
    )
    top_k = st.slider("Evidence chunks", 1, 4, 3)
    if st.button("Run traced pipeline", type="primary"):
        if not question.strip():
            st.warning("Enter a question before running the pipeline.")
            st.stop()
        lab = KnowledgeAugmentationLab(sample_documents(), scopes={"public"})
        result = lab.run(strategy or "advanced-rag", question, top_k=top_k)
        st.markdown("### Grounded answer")
        st.success(result.answer)
        score = lexical_groundedness(
            result.answer,
            " ".join(chunk.text for chunk in result.evidence),
            citation_ids=result.citations,
        )
        a, b, c = st.columns(3)
        a.metric("Groundedness", f"{score:.0%}")
        b.metric("Evidence chunks", len(result.evidence))
        c.metric("Citations", len(result.citations))
        st.markdown("### Decision trace")
        for index, step in enumerate(result.trace, start=1):
            st.markdown(
                render_trace_step_html(index, step),
                unsafe_allow_html=True,
            )
            if step.attributes:
                st.json(asdict(step)["attributes"])
        with st.expander("Inspect evidence"):
            for chunk in result.evidence:
                st.markdown(f"**{chunk.document_id}** · `{chunk.id}`")
                st.write(chunk.text)

elif page == "All-family showcase":
    st.subheader("Execute every local augmentation primitive")
    st.write(
        "One command exercises retrieval, context reuse, graph traversal, community reports, "
        "scoped memory, table provenance, and tool calls."
    )
    if st.button("Run all eight paths", type="primary"):
        output = run_showcase()
        st.toast("All augmentation paths completed", icon="✅")
        for name, details in output.items():
            with st.expander(name, expanded=name in {"advanced-rag", "kag"}):
                st.json(details)

else:
    st.subheader("Versioned lexical regression fixture")
    st.write(
        "The packaged core-retrieval-v1 fixture contains three self-authored public lexical cases. "
        "It detects deterministic regressions; it is not a benchmark or semantic-quality claim."
    )
    if st.button("Run packaged fixture", type="primary"):
        report = run_evaluation_fixture(load_evaluation_fixture())
        st.json(evaluation_report_dict(report))
    st.markdown("#### Wider production evaluation requirements")
    rows = [
        {
            "Layer": "Retrieval",
            "Current": "Recall/Precision@k · RR/MRR · binary nDCG",
            "Required next": "Held-out external and semantic strata",
        },
        {
            "Layer": "Context",
            "Current": "Selected IDs · lexical groundedness input",
            "Required next": "Coverage · relevance · redundancy · token budget",
        },
        {
            "Layer": "Answer",
            "Current": "Extractive citations · abstention",
            "Required next": "Correctness · entailment · citation precision/recall",
        },
        {
            "Layer": "Operations",
            "Current": "Row/path provenance · ToolResult",
            "Required next": "Plan accuracy · timeouts · side-effect audit",
        },
        {
            "Layer": "System",
            "Current": "Cache build/hit counters",
            "Required next": "p50/p95 latency · tokens · cost · reliability",
        },
        {
            "Layer": "Security",
            "Current": "RAG ACL ordering · boundary regressions",
            "Required next": "Poisoning · model injection · exfiltration",
        },
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.info(
        "Security scope: KnowledgeAugmentationLab filters source trust and ACLs before document indexing. "
        "Standalone graph, table, cache, memory, and tool primitives require caller-provided controls."
    )
    st.markdown("#### Catalog snapshot")
    st.json([asdict(concept) for concept in catalog[:3]])
