import json

import pytest

from labmate.core.traceview import render_trace, trace_view, write_trace_html


def span(sid, name, parent, start, ms, **attrs):
    return {
        "trace_id": attrs.pop("trace", "t1"),
        "span_id": sid,
        "parent_id": parent,
        "name": name,
        "start_ts": f"2026-09-26T10:00:{start:06.3f}+00:00",
        "latency_ms": ms,
        "status": attrs.pop("status", "ok"),
        **attrs,
    }


SPANS = [
    span("c1", "llm.chat", "s1", 1.0, 500, model="qwen3.6:35b-mlx", tokens_in=100, tokens_out=40),
    span("c2", "llm.chat", "s1", 1.5, 500, model="gemma4:26b-mlx", cached=True, tokens_in=5),
    span("s1", "step.extract", "r", 1.0, 1000, cards=6, rejected=1),
    span("i1", "llm.image", "s2", 2.0, 1000, model="x/z-image-turbo:latest"),
    span("s2", "step.cover", "r", 2.0, 1000, status="error", error="boom"),
    span("r", "run", None, 0.0, 4000, paper="2401.00001", mode="auto"),
]


def test_trace_view_lays_out_spans_on_one_axis():
    view = trace_view(SPANS)
    assert view.total_ms == 4000 and view.llm_calls == 3 and view.cached_calls == 1
    assert (view.tokens_in, view.tokens_out) == (105, 40)
    root, extract, chat = view.rows[:3]
    assert (root.kind, root.depth, root.left, root.width) == ("run", 0, 0, 100)
    assert (extract.label, extract.detail, extract.depth) == ("extract", "cards=6 rejected=1", 1)
    assert (chat.kind, chat.label, chat.depth, chat.left) == ("chat", "qwen3.6:35b-mlx", 2, 25)
    assert "100→40 tok" in chat.detail
    kinds = {r.label: r.kind for r in view.rows}
    assert kinds["x/z-image-turbo:latest"] == "image" and kinds["cover"] == "step"
    assert "error" in next(r for r in view.rows if r.label == "cover").attrs


def test_trace_view_needs_spans():
    with pytest.raises(ValueError):
        trace_view([])


def test_render_trace_escapes_and_splits_traces():
    spans = [*SPANS, span("x", "run", None, 0, 10, trace="t2", paper="<script>")]
    html = render_trace(spans, "Trace <b>")
    assert html.count('class="card trace"') == 2
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Trace &lt;b&gt;" in html
    assert "cached" in html and "error" in html


def test_write_trace_html(tmp_path):
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(s) + "\n" for s in SPANS))
    out = write_trace_html(tmp_path)
    assert out == tmp_path / "trace.html" and "extract" in out.read_text()
    assert "t1" in write_trace_html(tmp_path, all_traces=True, out=tmp_path / "a.html").read_text()
    with pytest.raises(FileNotFoundError):
        write_trace_html(tmp_path / "missing")
