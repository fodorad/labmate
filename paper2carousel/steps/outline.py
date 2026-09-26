"""Step 3 (ORCHESTRATOR): plan the slides and assign claim cards to them.

The slides are always the four blocks of a project page (task, challenges, proposed
method, main results); what goes into each depends on the paper, so the planner assigns
the claim cards and titles. Hard rules are enforced in code by :func:`check_outline` and
fed back to the model on violation.
"""

from __future__ import annotations

from collections import Counter

from paper2carousel.llm.structured import structured_chat
from paper2carousel.schemas import Claims, Outline, PaperType, Route
from paper2carousel.steps.llm import LLM, load_prompt

SECTIONS = ["task", "challenges", "method", "results"]
"""The four blocks every carousel has, in order (the structure of a project page)."""

LABELS = {
    "task": "Task",
    "challenges": "Challenges",
    "method": "Proposed method",
    "results": "Main results",
}
"""Display names of the blocks (slide badge, summary card title)."""

TEMPLATES: dict[PaperType, list[str]] = {
    t: SECTIONS for t in ("method", "benchmark", "survey", "position")
}
"""Slide purposes per paper type. All types share the four blocks; the paper type only
changes how the planner reads them (a benchmark's "method" is its design)."""

MAX_CLAIM_USES = 2
"""A claim may appear on at most this many slides."""


MAX_SLIDE_CLAIMS = 5
"""Claim cards one slide may be built on (enough material for four specific bullets)."""


def format_claims(claims: Claims) -> str:
    """One line per claim card, as shown to the planner and in the gate file.

    Args:
        claims: Claim cards.

    Returns:
        ``c01 [result] claim text (Section, p. 8)`` lines.
    """
    return "\n".join(
        f"{c.id} [{c.kind}] {c.claim} ({c.section}, p. {c.page})" for c in claims.cards
    )


def check_outline(outline: Outline, claims: Claims, paper_type: PaperType) -> list[str]:
    """Rules the outline must satisfy (a programmatic gate between chain steps).

    Args:
        outline: Candidate outline.
        claims: Available claim cards.
        paper_type: Routed paper type.

    Returns:
        Human-readable problems; empty if the outline is valid.
    """
    allowed = TEMPLATES[paper_type]
    known = {c.id for c in claims.cards}
    problems = []
    purposes = [slide.purpose for slide in outline.slides]
    if purposes != allowed:
        problems.append(f"the slides must be exactly {allowed}, in this order; got {purposes}")
    for i, slide in enumerate(outline.slides, start=1):
        if not 1 <= len(slide.claim_ids) <= MAX_SLIDE_CLAIMS:
            problems.append(
                f"slide {i}: needs 1 to {MAX_SLIDE_CLAIMS} claim ids, has {len(slide.claim_ids)}"
            )
        unknown = [cid for cid in slide.claim_ids if cid not in known]
        if unknown:
            problems.append(f"slide {i}: unknown claim ids {unknown}")
    overused = [
        cid
        for cid, n in Counter(cid for s in outline.slides for cid in s.claim_ids).items()
        if n > MAX_CLAIM_USES
    ]
    if overused:
        problems.append(f"claims used on more than {MAX_CLAIM_USES} slides: {sorted(overused)}")
    return problems


def plan_outline(
    title: str, route: Route, claims: Claims, llm: LLM, n_min: int = 6, n_max: int = 8
) -> Outline:
    """Plan the carousel from the claim cards.

    Args:
        title: Paper title.
        route: Routing decision (selects the template).
        claims: Verified claim cards.
        llm: Model settings.
        n_min: Minimum slides to ask for.
        n_max: Maximum slides to ask for.

    Returns:
        An outline that passes :func:`check_outline`.

    Raises:
        ValueError: If there are no claim cards to plan with.
    """
    if not claims.cards:
        raise ValueError("no verified claim cards: nothing to build slides from")
    allowed = TEMPLATES[route.paper_type]
    prompt = load_prompt("outline").format(
        title=title,
        paper_type=route.paper_type,
        n_min=n_min,
        n_max=n_max,
        purposes=", ".join(allowed),
        first=allowed[0],
        claims=format_claims(claims),
    )
    return structured_chat(
        llm.backend,
        llm.request(prompt),
        Outline,
        check=lambda o: check_outline(o, claims, route.paper_type),
    )
