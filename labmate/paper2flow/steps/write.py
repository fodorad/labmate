"""PROMPT CHAINING: write each planned card from its claim cards.

The outline is fixed at this point, so writing is a plain chain step: one call per card,
run in parallel since the cards don't depend on each other. Each bullet must cite the
claim ids it uses, and only its card's claims. That citation is what the fact-check loop
verifies next.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from labmate.core.factcheck import check_card, format_evidence
from labmate.core.lc import batch_map, prompt, structured
from labmate.paper2flow.prompts import load_prompt
from labmate.paper2flow.schemas import Card, Cards, ClaimCard, Claims, Outline, PlannedCard

PURPOSE_HINTS = {
    "task": "what the paper sets out to do, its inputs, outputs and setting.",
    "challenges": "what makes the problem hard and where earlier approaches fall short.",
    "method": "what the authors propose and how it works, component by component.",
    "results": "the main results with numbers and comparisons, ablations or limitations.",
}
"""What each card should say (the writer sees the one for its card)."""


def write_card(
    position: int,
    planned: PlannedCard,
    by_id: dict[str, ClaimCard],
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> Card:
    """Write one card from its outline entry (one validated model call).

    Args:
        position: 1-based card number.
        planned: The card's outline entry.
        by_id: Claim cards by id.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        The written card, with the planned title.
    """
    writer = structured(model, Card, check=lambda c: check_card(c, set(planned.claim_ids)))
    card = (prompt(load_prompt("write")) | writer).invoke(
        {
            "position": position,
            "purpose": planned.purpose,
            "purpose_hint": PURPOSE_HINTS.get(planned.purpose, ""),
            "title": planned.title,
            "claims": format_evidence([by_id[cid] for cid in planned.claim_ids]),
        },
        config,
    )
    return card.model_copy(update={"title": planned.title})


def write_cards(
    outline: Outline,
    claims: Claims,
    model: BaseChatModel,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> Cards:
    """Write all four cards of the outline, in parallel.

    Args:
        outline: The outline.
        claims: Claim cards.
        model: The writer model.
        workers: Concurrent writing calls.
        config: The calling step's config (callbacks).

    Returns:
        The written cards, in outline order.
    """
    by_id = {c.id: c for c in claims.cards}
    cards = batch_map(
        lambda item, cfg: write_card(item[0], item[1], by_id, model, cfg),
        list(enumerate(outline.cards, start=1)),
        config,
        workers,
    )
    return Cards(cards=cards)
