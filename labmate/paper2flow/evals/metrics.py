"""Run metrics computed from a finished run's artifacts. No model needed."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from labmate.paper2flow.chain import ARTIFACTS
from labmate.paper2flow.schemas import Claims, FactChecked, Flows, Outline, Paper

CARD_KINDS = {
    "task": {"task"},
    "challenges": {"challenge", "limitation"},
    "method": {"method", "contribution"},
    "results": {"result"},
}
"""Claim kinds that belong on each of the four cards."""


class RunMetrics(BaseModel):
    """Headline numbers for one run.

    Attributes:
        paper_id: Run directory name.
        title: Paper title.
        claims_verified: Claim cards that passed the quote check.
        claims_rejected: Claims dropped because their quote wasn't in the paper.
        bullets_first: Bullets in the writer's first draft.
        unsupported_first: First-draft bullets failing the fact-check.
        bullets_final: Bullets on the final cards.
        dropped: Bullets removed after the rewrite budget.
        rounds: Fact-check rounds used.
        card_fit: Share of final bullets that cite at least one claim of their card's
            kind (a task claim on Task, a result on Main results, ...): whether the
            orchestrator put the paper's claims in the right place.
        cards: Cards in the overview (four, unless one lost all its bullets).
        flow_nodes: Boxes in the end-to-end flow diagram.
        flow_details: Detail diagrams that break down its steps.
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
    card_fit: float
    cards: int
    flow_nodes: int
    flow_details: int

    @property
    def unsupported_first_pct(self) -> float:
        """Share of first-draft bullets that failed the fact-check, in percent."""
        return 100 * self.unsupported_first / self.bullets_first if self.bullets_first else 0.0


def card_fit(claims: Claims, outline: Outline, checked: FactChecked) -> float:
    """Share of final bullets citing a claim of their card's kind.

    Args:
        claims: Claim cards.
        outline: The outline (each card's purpose).
        checked: The final cards (a card that lost all bullets is not among them).

    Returns:
        The share, from 0 to 1 (0 without bullets).
    """
    kinds = {c.id: c.kind for c in claims.cards}
    dropped = set(checked.report.dropped_cards)
    kept = [p for i, p in enumerate(outline.cards, start=1) if i not in dropped]
    placed = [
        any(kinds.get(cid) in CARD_KINDS.get(planned.purpose, set()) for cid in b.claim_ids)
        for card, planned in zip(checked.cards.cards, kept, strict=True)
        for b in card.bullets
    ]
    return round(sum(placed) / len(placed), 3) if placed else 0.0


def run_metrics(run_dir: Path) -> RunMetrics:
    """Compute metrics for a finished run.

    Args:
        run_dir: The run directory (``runs/<paper id>``).

    Returns:
        The metrics.

    Raises:
        FileNotFoundError: If the run hasn't reached the fact-check step.
    """

    def load[M: BaseModel](field: str, model: type[M]) -> M:
        return model.model_validate_json((run_dir / ARTIFACTS[field]).read_text())

    claims, outline = load("claims", Claims), load("outline", Outline)
    checked = load("checked", FactChecked)
    flows = load("flows", Flows) if (run_dir / ARTIFACTS["flows"]).exists() else None
    report = checked.report
    return RunMetrics(
        paper_id=run_dir.name,
        title=load("paper", Paper).title,
        claims_verified=len(claims.cards),
        claims_rejected=len(claims.rejected),
        bullets_first=report.total_first,
        unsupported_first=report.failed_first,
        bullets_final=sum(len(c.bullets) for c in checked.cards.cards),
        dropped=len(report.dropped),
        rounds=len(report.rounds),
        card_fit=card_fit(claims, outline, checked),
        cards=len(checked.cards.cards),
        flow_nodes=len(flows.overview.nodes) if flows else 0,
        flow_details=len(flows.details) if flows else 0,
    )


def results_markdown(metrics: list[RunMetrics]) -> str:
    """Results table for the README.

    Args:
        metrics: One entry per run.

    Returns:
        Markdown with a per-paper table and a totals line.
    """
    rows = [
        "| Paper | Claims (verified / rejected) | Unsupported in first draft | "
        "Dropped after loop | Final bullets | Card fit | Flow diagrams |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        rows.append(
            f"| {m.title} | {m.claims_verified} / {m.claims_rejected} | "
            f"{m.unsupported_first}/{m.bullets_first} ({m.unsupported_first_pct:.0f}%) | "
            f"{m.dropped} | {m.bullets_final} | {100 * m.card_fit:.0f}% | "
            f"{m.flow_nodes} boxes + {m.flow_details} details |"
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
