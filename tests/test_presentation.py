from knowledge_aug_lab.models import TraceStep
from knowledge_aug_lab.presentation import render_trace_step_html


def test_trace_html_escapes_step_name_and_user_derived_detail() -> None:
    rendered = render_trace_step_html(
        1,
        TraceStep(
            '<script>alert("name")</script>',
            '<img src=x onerror="alert(1)"> & query',
        ),
    )

    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; query" in rendered
