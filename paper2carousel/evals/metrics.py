"""Run metrics computed from a finished run's artifacts and trace. No model needed.

Because the artifacts are reproducible from cassettes (``make replay``), so are these
numbers, which is what makes the published results checkable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from paper2carousel.schemas import Claims, FactChecked, Paper
from paper2carousel.tracing import read_trace

Span = dict[str, Any]


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
        contribution_coverage: Share of contribution claims used on some slide.
        slides: Slides in the final carousel.
        visuals: Slides with a figure or diagram.
        llm_calls: Model calls of the run.
        tokens_in: Prompt tokens.
        tokens_out: Generated tokens.
        swaps: Model swaps during the fact-check loop.
        wall_s: Compute seconds of the run (paused + approved invocations, without the
            time the outline waited for review).
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
    contribution_coverage: float
    slides: int
    visuals: int
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


def run_metrics(run_dir: Path) -> RunMetrics:
    """Compute metrics for a finished agentic run.

    Args:
        run_dir: The run directory (``runs/<paper_id>``).

    Returns:
        The metrics.

    Raises:
        FileNotFoundError: If the run hasn't reached the fact-check step.
    """
    paper = Paper.model_validate_json((run_dir / "00_paper.json").read_text())
    claims = Claims.model_validate_json((run_dir / "02_claims.json").read_text())
    checked = FactChecked.model_validate_json((run_dir / "05_factcheck.json").read_text())
    report, final = checked.report, checked.slides

    used = {cid for s in final.slides for b in s.bullets for cid in b.claim_ids}
    contributions = [c.id for c in claims.cards if c.kind == "contribution"]
    coverage = (
        sum(cid in used for cid in contributions) / len(contributions) if contributions else 0.0
    )
    visuals_file = run_dir / "06_visuals.json"
    visuals = (
        sum(v is not None for v in json.loads(visuals_file.read_text())["slides"])
        if visuals_file.exists()
        else 0
    )
    spans = latest_completed(run_dir / "trace.jsonl")
    llm = [s for s in spans if s["name"] in ("llm.chat", "llm.image")]
    roots = [s for s in spans if s["name"] == "run" and s["parent_id"] is None]
    return RunMetrics(
        paper_id=run_dir.name,
        title=paper.title,
        claims_verified=len(claims.cards),
        claims_rejected=len(claims.rejected),
        bullets_first=report.total_first,
        unsupported_first=report.failed_first,
        bullets_final=sum(len(s.bullets) for s in final.slides),
        dropped=len(report.dropped),
        rounds=len(report.rounds),
        contribution_coverage=round(coverage, 3),
        slides=len(final.slides),
        visuals=visuals,
        llm_calls=len(llm),
        tokens_in=sum(int(s.get("tokens_in") or 0) for s in llm),
        tokens_out=sum(int(s.get("tokens_out") or 0) for s in llm),
        swaps=sum(int(s.get("swaps") or 0) for s in spans if s["name"] == "step.factcheck"),
        wall_s=round(sum(float(r.get("latency_ms") or 0) for r in roots) / 1000, 1),
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
        "Dropped after loop | Final bullets | Contribution coverage | Visuals | LLM calls | "
        "Wall time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        rows.append(
            f"| {m.title} | {m.claims_verified} / {m.claims_rejected} | "
            f"{m.unsupported_first}/{m.bullets_first} ({m.unsupported_first_pct:.0f}%) | "
            f"{m.dropped} | {m.bullets_final} | {100 * m.contribution_coverage:.0f}% | "
            f"{m.visuals}/{m.slides} | {m.llm_calls} | {m.wall_s:.0f} s |"
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
