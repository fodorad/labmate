"""Run metrics computed from a finished run's artifacts and trace. No model needed.

Because the artifacts are reproducible from cassettes (``make replay``), so are these
numbers, which is what makes the published results checkable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from paper2carousel.schemas import Claims, FactChecked, MethodGraph, Outline
from paper2carousel.tracing import read_trace

Span = dict[str, Any]

BLOCK_KINDS = {
    "task": {"task"},
    "challenges": {"challenge", "limitation"},
    "method": {"method", "contribution"},
    "results": {"result"},
}
"""Claim kinds that belong in each of the four blocks."""


class RunMetrics(BaseModel):
    """Headline numbers for one run.

    Attributes:
        paper_id: Run directory name.
        title: Paper title.
        claims_verified: Claim cards that passed the quote check.
        claims_rejected: Claims dropped because their quote wasn't in the paper.
        bullets_first: Bullets in the writer's first draft.
        unsupported_first: First-draft bullets failing the fact-check.
        bullets_final: Bullets on the published slides.
        dropped: Bullets removed after the rewrite budget.
        rounds: Fact-check rounds used.
        block_fit: Share of final bullets that cite at least one claim of their block's
            kind (a task claim in Task, a result in Main results, ...): whether the
            orchestrator put the paper's claims in the right place.
        slides: Blocks in the summary (four).
        graph_nodes: Nodes of the post image's method graph (0 without a post).
        graph_edges: Edges of that graph.
        llm_calls: Distinct model requests the carousel needed.
        tokens_in: Prompt tokens of those requests.
        tokens_out: Generated tokens of those requests.
        swaps: Model swaps during the fact-check loop.
        wall_s: Seconds spent in those model calls when they ran live (excludes the time
            the outline waited for review, and replays).
    """

    paper_id: str
    title: str
    claims_verified: int
    claims_rejected: int
    bullets_first: int
    unsupported_first: int
    bullets_final: int
    dropped: int
    rounds: int
    block_fit: float
    slides: int
    graph_nodes: int
    graph_edges: int
    llm_calls: int
    tokens_in: int
    tokens_out: int
    swaps: int
    wall_s: float

    @property
    def unsupported_first_pct(self) -> float:
        """Share of first-draft bullets that failed the fact-check, in percent."""
        return 100 * self.unsupported_first / self.bullets_first if self.bullets_first else 0.0


def _is_live(chain: list[Span]) -> bool:
    return any(s["name"] in ("llm.chat", "llm.image") and not s.get("cached") for s in chain)


def latest_completed(trace: Path) -> list[Span]:
    """Spans of the latest run that produced a carousel, including the paused run before it.

    A paper is usually processed in two invocations: ``run`` pauses at the gate (root
    status ``awaiting_approval``) and ``run --approve`` finishes (root status ``ok``).
    Both belong to one logical run. Chains that made at least one live model call are
    preferred over pure replays, whose timings would be meaningless.

    Args:
        trace: ``trace.jsonl`` of a run directory.

    Returns:
        All spans of the chosen chain; empty if no run finished.
    """
    if not trace.exists():
        return []
    spans = read_trace(trace)
    roots = [s for s in spans if s["name"] == "run" and s["parent_id"] is None]
    chains: list[set[str]] = []
    for i, root in enumerate(roots):
        if root["status"] != "ok":
            continue
        ids = {str(root["trace_id"])}
        j = i - 1
        while j >= 0 and roots[j]["status"] == "awaiting_approval":
            ids.add(str(roots[j]["trace_id"]))
            j -= 1
        chains.append(ids)
    by_chain = [[s for s in spans if s["trace_id"] in ids] for ids in chains]
    live = [c for c in by_chain if _is_live(c)]
    return (live or by_chain or [[]])[-1]


def distinct_calls(spans: list[Span], cassettes: Path | None = None) -> list[Span]:
    """One span per distinct model request, preferring the live (uncached) call.

    A run is often several invocations (pause, approve, re-runs); a request answered live
    once and replayed later counts once, with its live latency. For a published gallery
    entry only the requests whose cassettes it ships (the ones replay needs) count.

    Args:
        spans: All spans of a run's trace.
        cassettes: The entry's cassette directory, if published.

    Returns:
        The chosen ``llm.chat`` / ``llm.image`` spans.
    """
    keep = (
        {p.stem for p in cassettes.glob("*/*.json")}
        if cassettes is not None and cassettes.exists()
        else None
    )
    chosen: dict[str, Span] = {}
    for s in spans:
        if s["name"] not in ("llm.chat", "llm.image"):
            continue
        key = str(s.get("key"))
        if keep is not None and key not in keep:
            continue
        if key not in chosen or (chosen[key].get("cached") and not s.get("cached")):
            chosen[key] = s
    return list(chosen.values())


def paper_title(run_dir: Path) -> str:
    """Paper title from ``00_paper.json``, or from ``meta.json`` for published gallery runs.

    Args:
        run_dir: Run directory or gallery entry.

    Returns:
        The title, or the directory name if neither file exists.
    """
    for name in ("00_paper.json", "meta.json"):
        path = run_dir / name
        if path.exists():
            return str(json.loads(path.read_text())["title"])
    return run_dir.name


def run_metrics(run_dir: Path) -> RunMetrics:
    """Compute metrics for a finished agentic run.

    Args:
        run_dir: The run directory (``runs/<paper_id>``) or a published gallery entry.

    Returns:
        The metrics.

    Raises:
        FileNotFoundError: If the run hasn't reached the fact-check step.
    """
    claims = Claims.model_validate_json((run_dir / "02_claims.json").read_text())
    checked = FactChecked.model_validate_json((run_dir / "05_factcheck.json").read_text())
    report, final = checked.report, checked.slides

    outline = Outline.model_validate_json((run_dir / "03_outline.json").read_text())
    kinds = {c.id: c.kind for c in claims.cards}
    placed = [
        any(kinds.get(cid) in BLOCK_KINDS.get(planned.purpose, set()) for cid in b.claim_ids)
        for slide, planned in zip(final.slides, outline.slides, strict=False)
        for b in slide.bullets
    ]
    graph_file = run_dir / "09_graph.json"
    graph = MethodGraph.model_validate_json(graph_file.read_text()) if graph_file.exists() else None
    trace = run_dir / "trace.jsonl"
    spans = read_trace(trace) if trace.exists() else []
    llm = distinct_calls(spans, run_dir / "cassettes")
    factchecks = [s for s in spans if s["name"] == "step.factcheck" and s.get("rounds")]
    return RunMetrics(
        paper_id=run_dir.name,
        title=paper_title(run_dir),
        claims_verified=len(claims.cards),
        claims_rejected=len(claims.rejected),
        bullets_first=report.total_first,
        unsupported_first=report.failed_first,
        bullets_final=sum(len(s.bullets) for s in final.slides),
        dropped=len(report.dropped),
        rounds=len(report.rounds),
        block_fit=round(sum(placed) / len(placed), 3) if placed else 0.0,
        slides=len(final.slides),
        graph_nodes=len(graph.nodes) if graph else 0,
        graph_edges=len(graph.edges) if graph else 0,
        llm_calls=len(llm),
        tokens_in=sum(int(s.get("tokens_in") or 0) for s in llm),
        tokens_out=sum(int(s.get("tokens_out") or 0) for s in llm),
        swaps=max((int(s.get("swaps") or 0) for s in factchecks), default=0),
        wall_s=round(
            sum(float(s.get("latency_ms") or 0) for s in llm if not s.get("cached")) / 1000, 1
        ),
    )


def results_markdown(metrics: list[RunMetrics]) -> str:
    """Results table for the README / gallery.

    Args:
        metrics: One entry per run.

    Returns:
        Markdown with a per-paper table and a totals line.
    """
    rows = [
        "| Paper | Claims (verified / rejected) | Unsupported in first draft | "
        "Dropped after loop | Final bullets | Block fit | Post graph | LLM calls | "
        "Wall time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        rows.append(
            f"| {m.title} | {m.claims_verified} / {m.claims_rejected} | "
            f"{m.unsupported_first}/{m.bullets_first} ({m.unsupported_first_pct:.0f}%) | "
            f"{m.dropped} | {m.bullets_final} | {100 * m.block_fit:.0f}% | "
            f"{m.graph_nodes} nodes, {m.graph_edges} edges | {m.llm_calls} | {m.wall_s:.0f} s |"
        )
    first = sum(m.bullets_first for m in metrics)
    unsupported = sum(m.unsupported_first for m in metrics)
    dropped = sum(m.dropped for m in metrics)
    if first:
        rows += [
            "",
            f"**Across {len(metrics)} paper(s):** {unsupported}/{first} first-draft bullets "
            f"({100 * unsupported / first:.0f}%) failed the fact-check; after the loop, "
            f"{dropped} were still unsupported and dropped, so every published bullet passed "
            "both the exact-number check and the judge.",
        ]
    return "\n".join(rows) + "\n"
