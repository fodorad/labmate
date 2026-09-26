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


class Theme(BaseModel):
    """Colours (hex) and font for the slides. M4 swaps in the adamfodor.com palette."""

    background: str = "#faf8f5"
    text: str = "#1c1c1e"
    muted: str = "#6b6b70"
    accent: str = "#2f6fdb"
    font: str = "Libertinus Serif"


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
        "theme": (theme or Theme()).model_dump(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Typst resolves image paths relative to the template, so it is compiled from the
    # output directory, where the figure and diagram PNGs live.
    source = out.with_suffix(".typ")
    source.write_text(files("paper2carousel.templates").joinpath("carousel.typ").read_text())
    pdf = typst.compile(str(source), root=str(out.parent), sys_inputs={"deck": json.dumps(data)})
    out.write_bytes(pdf)
    return out
