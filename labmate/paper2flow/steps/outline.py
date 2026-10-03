"""ORCHESTRATOR: plan the four cards and assign claim cards to them.

The overview always has the four cards of a project page (task, challenges, proposed
method, main results); what goes into each depends on the paper, so the planner assigns
the claim cards and titles. Hard rules are enforced in code by :func:`check_outline` and
fed back to the model on violation.
"""

from __future__ import annotations

from collections import Counter

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from labmate.core.lc import prompt, structured
from labmate.paper2flow.prompts import load_prompt
from labmate.paper2flow.schemas import Claims, Outline, Route

PURPOSES = ["task", "challenges", "method", "results"]
"""The four cards every overview has, in order (the structure of a project page)."""

LABELS = {
    "task": "Task",
    "challenges": "Challenges",
    "method": "Proposed method",
    "results": "Main results",
}
"""Display names of the cards."""

MAX_CLAIM_USES = 2
"""A claim may appear on at most this many cards."""

MAX_CARD_CLAIMS = 5
"""Claim cards one card may be built on (enough material for four specific bullets)."""


def format_claims(claims: Claims) -> str:
    """One line per claim card, as shown to the planner.

    Args:
        claims: Claim cards.

    Returns:
        ``c01 [result] claim text (Section, p. 8)`` lines.
    """
    return "\n".join(
        f"{c.id} [{c.kind}] {c.claim} ({c.section}, p. {c.page})" for c in claims.cards
    )


def check_outline(outline: Outline, claims: Claims) -> list[str]:
    """Rules the outline must satisfy (a programmatic gate between chain steps).

    Args:
        outline: Candidate outline.
        claims: Available claim cards.

    Returns:
        Human-readable problems; empty if the outline is valid.
    """
    known = {c.id for c in claims.cards}
    problems = []
    purposes = [card.purpose for card in outline.cards]
    if purposes != PURPOSES:
        problems.append(f"the cards must be exactly {PURPOSES}, in this order; got {purposes}")
    for i, card in enumerate(outline.cards, start=1):
        if not 1 <= len(card.claim_ids) <= MAX_CARD_CLAIMS:
            problems.append(
                f"card {i}: needs 1 to {MAX_CARD_CLAIMS} claim ids, has {len(card.claim_ids)}"
            )
        unknown = [cid for cid in card.claim_ids if cid not in known]
        if unknown:
            problems.append(f"card {i}: unknown claim ids {unknown}")
    overused = [
        cid
        for cid, n in Counter(cid for c in outline.cards for cid in c.claim_ids).items()
        if n > MAX_CLAIM_USES
    ]
    if overused:
        problems.append(f"claims used on more than {MAX_CLAIM_USES} cards: {sorted(overused)}")
    return problems


def plan_outline(
    title: str,
    route: Route,
    claims: Claims,
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> Outline:
    """Plan the four cards from the claim cards.

    Args:
        title: Paper title.
        route: What kind of paper it is.
        claims: Verified claim cards.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        An outline that passes :func:`check_outline`.

    Raises:
        ValueError: If there are no claim cards to plan with.
    """
    if not claims.cards:
        raise ValueError("no verified claim cards: nothing to build the overview from")
    planner = structured(model, Outline, check=lambda o: check_outline(o, claims))
    return (prompt(load_prompt("outline")) | planner).invoke(
        {"title": title, "paper_type": route.paper_type, "claims": format_claims(claims)},
        config,
    )
