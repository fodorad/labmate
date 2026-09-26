"""Plain-text summary with citations (extra output). Plain code, no LLM.

Every bullet is followed by the page and verbatim quote it rests on, which makes the
carousel auditable by anyone with the PDF.
"""

from __future__ import annotations

from paper2carousel.schemas import Claims, Paper, WrittenSlides


def summary_markdown(slides: WrittenSlides, claims: Claims, paper: Paper) -> str:
    """Build the summary.

    Args:
        slides: Final slides (with claim ids).
        claims: Claim cards (quotes and pages).
        paper: The paper.

    Returns:
        Markdown text.
    """
    by_id = {c.id: c for c in claims.cards}
    out = [f"# {slides.hook}", "", f"Summary of *{paper.title}*"]
    if paper.url:
        out[-1] += f" ({paper.url})"
    for i, slide in enumerate(slides.slides, start=1):
        out += ["", f"## {i}. {slide.title}", ""]
        for bullet in slide.bullets:
            out.append(f"- {bullet.text}")
            for cid in bullet.claim_ids:
                if cid in by_id:
                    c = by_id[cid]
                    out.append(f'  - p. {c.page} ({c.section}): "{c.evidence_quote}"')
    return "\n".join(out) + "\n"
