"""PROMPT CHAINING: group the claim cards into the four cards and write each one.

The four cards of a project page (task, challenges, proposed method, main results) are fixed,
and a claim card already says which of them it belongs to (its kind), so the cards are planned
in code: :func:`plan_cards`. Writing is then a plain chain step: one call per card, run in
parallel since the cards don't depend on each other. Each bullet must cite the claim ids it
uses, and only its card's claims. That citation is what the fact-check loop verifies next.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from labmate.core.factcheck import check_card, format_evidence
from labmate.core.lc import batch_map, prompt, structured
from labmate.paper2flow.prompts import load_prompt
from labmate.paper2flow.schemas import Card, Cards, ClaimCard, Claims, Outline, PlannedCard

PURPOSES = ["task", "challenges", "method", "results"]
"""The four cards every overview has, in order (the structure of a project page)."""

LABELS = {
    "task": "Task",
    "challenges": "Challenges",
    "method": "Proposed method",
    "results": "Main results",
}
"""Display names of the cards."""

CARD_KINDS = {
    "task": {"task"},
    "challenges": {"challenge", "limitation"},
    "method": {"method", "contribution"},
    "results": {"result"},
}
"""Claim kinds that belong on each card."""

MAX_CARD_CLAIMS = 5
"""Claim cards one card is built on (enough material for four specific bullets)."""

PURPOSE_HINTS = {
    "task": "what the paper sets out to do, its inputs, outputs and setting.",
    "challenges": "what makes the problem hard and where earlier approaches fall short.",
    "method": "what the authors propose and how it works, component by component.",
    "results": "the main results: every bullet names the task or dataset its number belongs to; "
    "then ablations or limitations.",
}
"""What each card should say (the writer sees the one for its card)."""


def plan_cards(claims: Claims) -> Outline:
    """Group the claim cards into the cards of the overview by their kind.

    Args:
        claims: Verified claim cards.

    Returns:
        The cards that have at least one claim, in the order of :data:`PURPOSES`, each with up
        to :data:`MAX_CARD_CLAIMS` claims in reading order.

    Raises:
        ValueError: If no claim card fits any card.
    """
    planned = []
    for purpose in PURPOSES:
        ids = [c.id for c in claims.cards if c.kind in CARD_KINDS[purpose]][:MAX_CARD_CLAIMS]
        if ids:
            planned.append(PlannedCard(purpose=purpose, claim_ids=ids))
    if not planned:
        raise ValueError("no verified claim cards: nothing to build the overview from")
    return Outline(cards=planned)


def write_card(
    planned: PlannedCard,
    by_id: dict[str, ClaimCard],
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> Card:
    """Write one card from its planned claims (one validated model call).

    Args:
        planned: The card's purpose and claims.
        by_id: Claim cards by id.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        The written card: a title and its bullets.
    """
    writer = structured(model, Card, check=lambda c: check_card(c, set(planned.claim_ids)))
    return (prompt(load_prompt("write")) | writer).invoke(
        {
            "label": LABELS[planned.purpose],
            "purpose_hint": PURPOSE_HINTS.get(planned.purpose, ""),
            "claims": format_evidence([by_id[cid] for cid in planned.claim_ids]),
        },
        config,
    )


def write_cards(
    outline: Outline,
    claims: Claims,
    model: BaseChatModel,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> Cards:
    """Write all cards of the outline, in parallel.

    Args:
        outline: The planned cards.
        claims: Claim cards.
        model: The writer model.
        workers: Concurrent writing calls.
        config: The calling step's config (callbacks).

    Returns:
        The written cards, in outline order.
    """
    by_id = {c.id: c for c in claims.cards}
    cards = batch_map(
        lambda planned, cfg: write_card(planned, by_id, model, cfg),
        list(outline.cards),
        config,
        workers,
    )
    return Cards(cards=cards)
