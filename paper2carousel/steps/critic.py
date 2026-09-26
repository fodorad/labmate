"""Slide critic (vision): review every rendered page, write alt texts, drop bad visuals.

A small non-MLX vision model looks at each page at phone resolution. Its alt texts are an
output in their own right; its verdicts gate the visuals: a figure that is illegible or
doesn't match the slide is removed (a deterministic remedy, so the loop can't oscillate).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pymupdf

from paper2carousel.llm.structured import structured_chat
from paper2carousel.llm.types import Message
from paper2carousel.parallel import parallel_map
from paper2carousel.schemas import Deck, Review, SlideReview
from paper2carousel.steps.llm import LLM, load_prompt

REVIEW_DPI = 72
"""540x675 px: roughly how a carousel page appears on a phone, and small cassettes."""


def page_pngs(pdf: Path, out_dir: Path, dpi: int = REVIEW_DPI) -> list[Path]:
    """Render every page of the carousel to PNG.

    Args:
        pdf: Carousel PDF.
        out_dir: Directory for ``page-01.png``, ...
        dpi: Render resolution.

    Returns:
        PNG paths in page order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            path = out_dir / f"page-{i:02d}.png"
            page.get_pixmap(dpi=dpi).save(path)
            paths.append(path)
    return paths


def _content(deck: Deck, page: int) -> str:
    if page == 1:
        return f"Cover. Headline: {deck.title}" + (
            " (with an illustration)" if deck.cover_image else ""
        )
    slide = deck.slides[page - 2]
    text = f"Title: {slide.title}\n" + "\n".join(f"- {b}" for b in slide.bullets)
    if slide.image:
        text += f"\nImage: {slide.image_caption or 'a figure'}"
    return text


def review_deck(deck: Deck, pdf: Path, vision: LLM, workers: int = 2) -> Review:
    """Review all pages and decide which visuals to drop.

    Args:
        deck: The rendered deck (for the intended content of each page).
        pdf: The rendered carousel.
        vision: Vision model settings.
        workers: Concurrent review calls.

    Returns:
        One review per page (cover first) and the 1-based content-slide numbers whose
        visual should be removed.
    """
    pngs = page_pngs(pdf, pdf.parent / "pages")
    template = load_prompt("critic")

    def review(item: tuple[int, Path]) -> SlideReview:
        page, png = item
        prompt = template.format(page=page, total=len(pngs), content=_content(deck, page))
        request = vision.request(prompt)
        image = base64.b64encode(png.read_bytes()).decode()
        request = request.model_copy(
            update={"messages": [Message(role="user", content=prompt, images=[image])]}
        )
        return structured_chat(vision.backend, request, SlideReview)

    pages = parallel_map(review, list(enumerate(pngs, start=1)), workers)
    dropped = [
        i
        for i, (slide, r) in enumerate(zip(deck.slides, pages[1:], strict=True), start=1)
        if slide.image and not (r.legible and r.visual_relevant)
    ]
    return Review(pages=pages, dropped_visuals=dropped)
