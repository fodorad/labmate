"""Step 8: render a deck to a PDF carousel with Typst. Plain code, no LLM.

Image paths in the deck are relative to the output PDF's directory.

Layout is deterministic and lives in ``templates/carousel.typ``; the model only ever
produces data. Content is passed as JSON through Typst's ``sys.inputs``, so paper text
cannot inject markup.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import typst
from pydantic import BaseModel

from paper2carousel.schemas import Deck, Paper

FONTS_DIR = files("paper2carousel.templates").joinpath("fonts")
"""Bundled fonts (Inter, SIL OFL 1.1), so renders don't depend on the machine's fonts."""


class Theme(BaseModel):
    """Colours (hex), font and handle. Defaults follow the adamfodor.com light theme."""

    background: str = "#f7f6f4"
    surface: str = "#fffefd"
    border: str = "#dcd8d2"
    text: str = "#222b35"
    muted: str = "#4c6176"
    accent: str = "#ab4c31"
    accent_muted: str = "#f1e1dc"
    font: str = "Inter"
    handle: str = "adamfodor.com"


def author_line(authors: list[str], limit: int = 4) -> str:
    """Authors for display: all of them if few, else the first few and "et al.".

    Args:
        authors: Author names.
        limit: Most names to show.

    Returns:
        A comma-separated line (empty if there are no authors).
    """
    shown = ", ".join(authors[:limit])
    return f"{shown} et al." if len(authors) > limit else shown


def render_deck(deck: Deck, paper: Paper, out: Path, theme: Theme | None = None) -> Path:
    """Render ``deck`` to a PDF (cover + one page per slide).

    Args:
        deck: Slides to render.
        paper: Source paper (title and link go on the slides).
        out: Output PDF path.
        theme: Colours and font.

    Returns:
        ``out``.
    """
    data = {
        "title": deck.title,
        "paper_title": paper.title,
        "authors": author_line(paper.authors),
        "source": paper.url or paper.title,
        "slides": [s.model_dump(exclude_none=True) for s in deck.slides],
        **({"cover_image": deck.cover_image} if deck.cover_image else {}),
        "theme": (theme or Theme()).model_dump(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Typst resolves image paths relative to the template, so it is compiled from the
    # output directory, where the figure and diagram PNGs live.
    source = out.with_suffix(".typ")
    source.write_text(files("paper2carousel.templates").joinpath("carousel.typ").read_text())
    pdf = typst.compile(
        str(source),
        root=str(out.parent),
        font_paths=[str(FONTS_DIR)],
        sys_inputs={"deck": json.dumps(data)},
    )
    out.write_bytes(pdf)
    return out


CARD_COLORS = {
    "Task": "#fff8d5",
    "Challenges": "#ffe8e7",
    "Proposed method": "#e3f2f8",
    "Main results": "#e4f8d6",
}
"""Card backgrounds of the project pages on adamfodor.com."""


def main_image(deck: Deck) -> tuple[str | None, str]:
    """The image for the summary's header: a paper figure, preferably the method slide's.

    Args:
        deck: Rendered deck (with visuals and labels).

    Returns:
        ``(path, caption)``; the path is relative to the run directory, ``None`` if the
        deck has no image at all (the cover illustration is used as a last resort).
    """
    method = [s for s in deck.slides if s.label == "Proposed method"]
    ordered = method + [s for s in deck.slides if s not in method]
    for figures_only in (True, False):
        for slide in ordered:
            if slide.image and (not figures_only or slide.image.startswith("figures/")):
                return slide.image, slide.image_caption or ""
    return deck.cover_image, ""


def render_summary(
    deck: Deck,
    paper: Paper,
    out: Path,
    theme: Theme | None = None,
    image: str | None = None,
    caption: str = "",
) -> Path:
    """Render the one-page summary (a project page as PDF) from the fact-checked deck.

    Args:
        deck: The final deck: its labelled slides become the cards.
        paper: Source paper (title, authors, abstract, link).
        out: Output PDF path; image paths are relative to its directory.
        theme: Colours and font.
        image: Header image (relative to the output directory); defaults to
            :func:`main_image` of the deck.
        caption: Caption of ``image``.

    Returns:
        ``out``.
    """
    if image is None:
        image, caption = main_image(deck)
    data = {
        "title": paper.title,
        "authors": ", ".join(paper.authors),
        "url": paper.url,
        "abstract": paper.abstract,
        "cards": [
            {
                "label": s.label or s.title,
                "title": s.title if s.label else "",
                "bullets": s.bullets,
                "color": CARD_COLORS.get(s.label or "", "#f1e1dc"),
            }
            for s in deck.slides
        ],
        "theme": (theme or Theme()).model_dump(),
        **({"image": image, "image_caption": caption} if image else {}),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    source = out.with_suffix(".typ")
    source.write_text(files("paper2carousel.templates").joinpath("summary.typ").read_text())
    pdf = typst.compile(
        str(source),
        root=str(out.parent),
        font_paths=[str(FONTS_DIR)],
        sys_inputs={"summary": json.dumps(data)},
    )
    out.write_bytes(pdf)
    return out
