"""Static HTML trace viewer: a waterfall of the spans in ``trace.jsonl``.

No server and no JavaScript framework: one self-contained HTML file per run, which also
ships in the gallery. Each row is a span (run, step or model call) placed on a shared
time axis; clicking a row shows all of its attributes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import ChoiceLoader, Environment, PackageLoader, select_autoescape
from pydantic import BaseModel

from labmate.core.theme import Theme
from labmate.core.tracing import latest_completed, read_trace

HIDDEN = {"trace_id", "span_id", "parent_id", "name", "start_ts", "latency_ms", "status"}
"""Span fields shown in the row itself rather than in the attribute list."""

INLINE = {
    "step.ingest": ["sections", "figures"],
    "step.route": ["paper_type", "confidence"],
    "step.extract": ["cards", "rejected"],
    "step.outline": ["slides"],
    "step.write": ["slides"],
    "step.factcheck": ["rounds", "failed_first", "total_first", "dropped", "swaps"],
    "step.post": ["takeaways", "dropped"],
    "step.visuals": ["tool_calls"],
    "step.critic": ["dropped_visuals"],
    "run": ["paper", "mode"],
}
"""Attributes summarised next to the span name, per span name."""


class Row(BaseModel):
    """One span, laid out for the waterfall.

    Attributes:
        kind: ``run``, ``step``, ``chat`` or ``image`` (drives the colour).
        label: Short name, e.g. ``factcheck`` or the model tag.
        detail: Key attributes inline, e.g. ``cards=6 rejected=0``.
        depth: Nesting level (0 = root).
        left: Start offset, percent of the trace duration.
        width: Duration, percent of the trace duration.
        ms: Duration in milliseconds.
        status: Span status (``ok``, ``error``, ``awaiting_approval``, ...).
        cached: True if a model call was served from a cassette.
        attrs: Remaining attributes, shown on click.
    """

    kind: str
    label: str
    detail: str
    depth: int
    left: float
    width: float
    ms: float
    status: str
    cached: bool
    attrs: dict[str, str]


class TraceView(BaseModel):
    """One trace (one invocation of the pipeline), ready to render.

    Attributes:
        trace_id: Trace id.
        started: Start time (ISO, UTC).
        total_ms: Wall-clock duration.
        llm_calls: Model calls.
        cached_calls: Model calls served from cassettes.
        tokens_in: Prompt tokens.
        tokens_out: Generated tokens.
        rows: Spans in start order.
    """

    trace_id: str
    started: str
    total_ms: float
    llm_calls: int
    cached_calls: int
    tokens_in: int
    tokens_out: int
    rows: list[Row]


def _kind(name: str) -> str:
    if name == "run":
        return "run"
    if name.startswith("step."):
        return "step"
    return "image" if name == "llm.image" else "chat"


def _label(span: dict[str, Any]) -> str:
    name = str(span["name"])
    if name.startswith("llm."):
        return str(span.get("model", name))
    return name.removeprefix("step.")


def _detail(span: dict[str, Any]) -> str:
    keys = INLINE.get(str(span["name"]), [])
    parts = [f"{k}={span[k]}" for k in keys if k in span and span[k] not in (None, [], "")]
    if span["name"] == "llm.chat" and span.get("tokens_out"):
        parts.append(f"{span.get('tokens_in', 0)}→{span['tokens_out']} tok")
    return " ".join(parts)


def trace_view(spans: list[dict[str, Any]]) -> TraceView:
    """Lay out the spans of one trace on a time axis.

    Args:
        spans: Spans sharing one ``trace_id``.

    Returns:
        The laid-out trace.

    Raises:
        ValueError: If ``spans`` is empty.
    """
    if not spans:
        raise ValueError("no spans")
    starts = {s["span_id"]: datetime.fromisoformat(s["start_ts"]) for s in spans}
    t0 = min(starts.values())
    offsets = {k: (v - t0).total_seconds() * 1000 for k, v in starts.items()}
    total = max(offsets[s["span_id"]] + float(s["latency_ms"]) for s in spans) or 1.0
    parents = {s["span_id"]: s["parent_id"] for s in spans}

    def depth(span_id: str) -> int:
        d, parent = 0, parents.get(span_id)
        while parent in parents:
            d, parent = d + 1, parents[parent]
        return d

    rows = [
        Row(
            kind=_kind(str(s["name"])),
            label=_label(s),
            detail=_detail(s),
            depth=depth(s["span_id"]),
            left=round(100 * offsets[s["span_id"]] / total, 3),
            width=round(max(100 * float(s["latency_ms"]) / total, 0.25), 3),
            ms=float(s["latency_ms"]),
            status=str(s["status"]),
            cached=bool(s.get("cached")),
            attrs={k: str(v) for k, v in s.items() if k not in HIDDEN},
        )
        for s in sorted(spans, key=lambda s: (offsets[s["span_id"]], depth(s["span_id"])))
    ]
    llm = [s for s in spans if str(s["name"]).startswith("llm.")]
    return TraceView(
        trace_id=str(spans[0]["trace_id"]),
        started=t0.isoformat(timespec="seconds"),
        total_ms=round(total, 1),
        llm_calls=len(llm),
        cached_calls=sum(bool(s.get("cached")) for s in llm),
        tokens_in=sum(int(s.get("tokens_in") or 0) for s in llm),
        tokens_out=sum(int(s.get("tokens_out") or 0) for s in llm),
        rows=rows,
    )


def environment(*packages: str) -> Environment:
    """Jinja environment over the shared HTML templates (plus a feature's), with autoescaping.

    Args:
        packages: Feature packages whose ``templates`` directory is searched first, e.g.
            ``"labmate.paper2flow"``.

    Returns:
        The environment.
    """
    loaders = [PackageLoader(p, "templates") for p in packages]
    return Environment(
        loader=ChoiceLoader([*loaders, PackageLoader("labmate.core", "templates")]),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_trace(spans: list[dict[str, Any]], title: str, theme: Theme | None = None) -> str:
    """Render spans (one or more traces) as a standalone HTML page.

    Args:
        spans: Spans, possibly from several traces (e.g. paused + approved run).
        title: Page title.
        theme: Colours.

    Returns:
        HTML.
    """
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for s in spans:
        by_trace.setdefault(str(s["trace_id"]), []).append(s)
    traces = [trace_view(group) for group in by_trace.values()]
    template = environment().get_template("trace.html.j2")
    return template.render(title=title, traces=traces, theme=theme or Theme())


def write_trace_html(run_dir: Path, all_traces: bool = False, out: Path | None = None) -> Path:
    """Write ``trace.html`` for a run.

    Args:
        run_dir: Run directory with ``trace.jsonl``.
        all_traces: Every invocation instead of the latest completed run.
        out: Output path (default ``run_dir/trace.html``).

    Returns:
        The written path.

    Raises:
        FileNotFoundError: If the run has no trace.
    """
    trace = run_dir / "trace.jsonl"
    if not trace.exists():
        raise FileNotFoundError(trace)
    spans = read_trace(trace) if all_traces else latest_completed(trace) or read_trace(trace)
    out = out or run_dir / "trace.html"
    out.write_text(render_trace(spans, f"Trace · {run_dir.name}"))
    return out
