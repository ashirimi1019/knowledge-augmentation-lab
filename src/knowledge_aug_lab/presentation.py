"""Safe presentation helpers for the visual explorer."""

from __future__ import annotations

from html import escape

from knowledge_aug_lab.models import TraceStep


def render_trace_step_html(index: int, step: TraceStep) -> str:
    """Render one trace card while treating all step content as plain text."""

    name = escape(step.name, quote=True)
    detail = escape(step.detail, quote=True)
    return f'<div class="trace"><b>{index:02d} · {name}</b><br><span class="small-note">{detail}</span></div>'
