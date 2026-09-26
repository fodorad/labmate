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
