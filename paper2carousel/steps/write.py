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

MAX_WORDS = 25
"""Hard limit per bullet (the prompt asks for 20; a little slack avoids needless retries)."""


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


def write_slides(outline: Outline, claims: Claims, llm: LLM, workers: int = 2) -> WrittenSlides:
    """Write all slides of an approved outline.

    Args:
        outline: Approved outline.
        claims: Claim cards.
        llm: Model settings.
        workers: Concurrent writing calls.

    Returns:
        The written slides, in outline order.
    """
    by_id = {c.id: c for c in claims.cards}
    template = load_prompt("write")
    total = len(outline.slides)

    def write(item: tuple[int, OutlineSlide]) -> SlideText:
        position, planned = item
        cards = [by_id[cid] for cid in planned.claim_ids]
        prompt = template.format(
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

    slides = parallel_map(write, list(enumerate(outline.slides, start=1)), workers)
    return WrittenSlides(hook=outline.hook, slides=slides)
