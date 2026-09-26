"""Step 5 (PROMPT CHAINING): write each planned slide from its claim cards.

The outline is fixed at this point, so writing is a plain chain: one call per slide,
run in parallel since slides don't depend on each other. Each bullet must cite the claim
ids it uses, and only the slide's own claims. That citation is what the M3 fact-check
loop verifies.
"""

from __future__ import annotations

from paper2carousel.llm.structured import structured_chat
from paper2carousel.parallel import parallel_map
from paper2carousel.schemas import (
    ClaimCard,
    Claims,
    Outline,
    OutlineSlide,
    SlideText,
    WrittenSlides,
)
from paper2carousel.steps.llm import LLM, load_prompt

MAX_WORDS = 30
"""Hard limit per bullet (the prompt asks for 25; a little slack avoids needless retries)."""


def check_slide(slide: SlideText, allowed_ids: set[str]) -> list[str]:
    """Rules for a written slide.

    Args:
        slide: Candidate slide.
        allowed_ids: Claim ids assigned to this slide by the outline.

    Returns:
        Problems; empty if valid.
    """
    problems = []
    if len(slide.title.split()) > 10:
        problems.append("the title has more than 10 words")
    for i, bullet in enumerate(slide.bullets, start=1):
        if len(bullet.text.split()) > MAX_WORDS:
            problems.append(f"bullet {i} has more than {MAX_WORDS} words")
        foreign = sorted(set(bullet.claim_ids) - allowed_ids)
        if foreign:
            problems.append(f"bullet {i} cites claims not on this slide: {foreign}")
    return problems


def format_cards(cards: list[ClaimCard]) -> str:
    """Claim cards with their evidence, as shown to the writer.

    Args:
        cards: The slide's claim cards.

    Returns:
        Text block.
    """
    return "\n".join(
        f'{c.id} [{c.kind}] {c.claim}\n    evidence: "{c.evidence_quote}"' for c in cards
    )


def write_slide(
    position: int, planned: OutlineSlide, total: int, by_id: dict[str, ClaimCard], llm: LLM
) -> SlideText:
    """Write one slide from its outline entry (one model call, validated).

    Args:
        position: 1-based slide number.
        planned: The slide's outline entry.
        total: Number of slides.
        by_id: Claim cards by id.
        llm: Model settings.

    Returns:
        The written slide.
    """
    cards = [by_id[cid] for cid in planned.claim_ids]
    prompt = load_prompt("write").format(
        position=position,
        total=total,
        purpose=planned.purpose,
        title=planned.title,
        claims=format_cards(cards),
    )
    return structured_chat(
        llm.backend,
        llm.request(prompt),
        SlideText,
        check=lambda s: check_slide(s, set(planned.claim_ids)),
    )


def write_slides(outline: Outline, claims: Claims, llm: LLM, workers: int = 2) -> WrittenSlides:
    """Write all slides of an approved outline (parallel over slides).

    Args:
        outline: Approved outline.
        claims: Claim cards.
        llm: Model settings.
        workers: Concurrent writing calls.

    Returns:
        The written slides, in outline order.
    """
    by_id = {c.id: c for c in claims.cards}
    total = len(outline.slides)
    slides = parallel_map(
        lambda item: write_slide(item[0], item[1], total, by_id, llm),
        list(enumerate(outline.slides, start=1)),
        workers,
    )
    return WrittenSlides(hook=outline.hook, slides=slides)
