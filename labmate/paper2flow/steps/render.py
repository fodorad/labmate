"""``overview.pdf``, rendered with Typst. Plain code, no LLM.

The overview (A4 portrait) is the paper at a glance: a cover (title, authors, venue and
date, the main figure), the four cards, the end-to-end data flow, and one page per
detail flow. paper2post reuses the helpers here for ``post.pdf``.

Layout is deterministic and lives in ``templates/*.typ``; the model only ever produces
data. Content is passed as JSON through Typst's ``sys.inputs``, so paper text cannot
inject markup. No PDF carries a creation date, so the same run renders the same bytes.
"""

from __future__ import annotations

import json
import struct
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

import typst

from labmate.core.figures import best_figure, short_caption
from labmate.core.theme import FONTS_DIR, Theme
from labmate.paper2flow.schemas import Cards, Flows, Paper
from labmate.paper2flow.steps.flow import LETTERS, SCALE, legend
from labmate.paper2flow.steps.write import LABELS

CARD_COLORS = {
    "Task": "#fff8d5",
    "Challenges": "#ffe8e7",
    "Proposed method": "#e3f2f8",
    "Main results": "#e4f8d6",
}
"""Card backgrounds of the project pages on adamfodor.com."""

FLOW_KICKER = "End-to-end data flow"
"""Heading of the overview diagram."""


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


def _pretty_date(text: str) -> str:
    """``2017-06-12`` becomes ``12 June 2017``; anything else is kept as written."""
    try:
        d = date.fromisoformat(text)
    except ValueError:
        return text
    return f"{d.day} {d:%B %Y}"


def publication_line(paper: Paper) -> str:
    """Where and when the paper appeared, e.g. ``J. Imaging 2023, 9, 196 · 21 September 2023``.

    Args:
        paper: The paper (venue, date, year, url).

    Returns:
        The line; ``arXiv preprint`` stands in for a missing venue of an arXiv paper, the
        year for a missing date. Empty if nothing is known.
    """
    venue = paper.venue or ("arXiv preprint" if "arxiv.org" in paper.url else "")
    when = _pretty_date(paper.date) if paper.date else ""
    if not when and paper.year and str(paper.year) not in venue:
        when = str(paper.year)
    return " · ".join(part for part in (venue, when) if part)


def card_blocks(cards: Cards, labels: list[str]) -> list[dict[str, Any]]:
    """The four cards as the template draws them (citations dropped).

    Args:
        cards: Fact-checked cards.
        labels: One label per card ("Task", ...).

    Returns:
        ``{"label", "title", "bullets", "color"}`` per card.
    """
    return [
        {
            "label": label,
            "title": c.title,
            "bullets": [b.text for b in c.bullets],
            "color": CARD_COLORS.get(label, "#f1e1dc"),
        }
        for c, label in zip(cards.cards, labels, strict=True)
    ]


def main_figure(cards: Cards, labels: list[str], paper: Paper) -> tuple[str, str] | None:
    """The paper figure for the cover: the one matching the method card best, else Figure 1.

    Args:
        cards: Fact-checked cards.
        labels: Their labels.
        paper: The paper (its figures).

    Returns:
        ``(path relative to the run directory, caption)``, or ``None`` without figures.
    """
    method = next(
        (c for c, label in zip(cards.cards, labels, strict=True) if label == LABELS["method"]),
        None,
    )
    text = " ".join([method.title, *(b.text for b in method.bullets)]) if method else ""
    figure = best_figure(text, paper.figures, paper.title) or next(iter(paper.figures), None)
    return (figure.path, short_caption(figure.caption)) if figure else None


def natural_size(png: Path) -> tuple[float, float]:
    """Size in points of a diagram PNG at its render scale (read from the PNG header).

    A small diagram is shown at this size at most, so its boxes aren't blown up to fill
    the page.

    Args:
        png: PNG file rendered at :data:`~labmate.paper2flow.steps.flow.SCALE`.

    Returns:
        ``(width, height)`` in points (CSS pixels at 96 per inch, 72 points per inch).
    """
    width, height = struct.unpack(">II", png.read_bytes()[16:24])
    return width / SCALE * 0.75, height / SCALE * 0.75


def diagrams(flows: Flows, images: list[str], image_dir: Path) -> list[dict[str, Any]]:
    """One page description per diagram: kicker, title, caption, image and legend.

    Args:
        flows: All diagrams.
        images: Their PNG file names (overview first).
        image_dir: Where the PNGs are (the output directory).

    Returns:
        Page data, overview first.
    """
    labels = {n.id: n.label for n in flows.overview.nodes}
    kickers = [FLOW_KICKER] + [
        f"Detail {LETTERS[i]} · {labels.get(d.node_id, d.node_id)}"
        for i, d in enumerate(flows.details)
    ]
    graphs = [flows.overview, *(d.graph for d in flows.details)]
    pages = []
    for kicker, graph, image in zip(kickers, graphs, images, strict=True):
        width, height = natural_size(image_dir / image)
        pages.append(
            {
                "kicker": kicker,
                "title": graph.title,
                "caption": graph.caption,
                "image": image,
                "width": round(width, 1),
                "height": round(height, 1),
                "legend": legend(graph),
            }
        )
    return pages


def compile_typst(package: str, template: str, data: dict[str, Any], out: Path) -> Path:
    """Compile a Typst template shipped in ``package`` with ``data`` into ``out``.

    Typst resolves image paths relative to the source file, so the template is copied
    next to the output (where the images are) for the compile and removed afterwards.

    Args:
        package: Package holding the template, e.g. ``"labmate.paper2flow.templates"``.
        template: File name, e.g. ``"overview.typ"``.
        data: Content, passed as JSON through ``sys.inputs``.
        out: Output PDF path.

    Returns:
        ``out``.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    source = out.with_suffix(".typ")
    source.write_text(files(package).joinpath(template).read_text())
    try:
        pdf = typst.compile(
            str(source),
            root=str(out.parent),
            font_paths=[str(FONTS_DIR)],
            sys_inputs={"data": json.dumps(data)},
        )
    finally:
        source.unlink(missing_ok=True)
    out.write_bytes(pdf)
    return out


def render_overview(
    paper: Paper,
    cards: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    out: Path,
    figure: tuple[str, str] | None = None,
    theme: Theme | None = None,
) -> Path:
    """Render ``overview.pdf``.

    Args:
        paper: The paper (title, authors, venue, date, link, abstract).
        cards: The four cards (:func:`card_blocks`).
        pages: The diagrams (:func:`diagrams`).
        out: Output PDF path; image paths are relative to its directory.
        figure: Main figure ``(path, caption)``; without one, page 1 shows the abstract.
        theme: Colours and font.

    Returns:
        ``out``.
    """
    data = {
        "title": paper.title,
        "authors": ", ".join(paper.authors),
        "publication": publication_line(paper),
        "url": paper.url,
        "abstract": paper.abstract,
        "cards": cards,
        "diagrams": pages,
        "theme": (theme or Theme()).model_dump(),
        **({"figure": figure[0], "figure_caption": figure[1]} if figure else {}),
    }
    return compile_typst("labmate.paper2flow.templates", "overview.typ", data, out)
