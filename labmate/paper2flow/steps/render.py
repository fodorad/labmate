"""The two outputs, rendered with Typst. Plain code, no LLM.

- ``overview.pdf`` (A4 portrait): the paper at a glance: title, authors, venue and date,
  the main figure; the four blocks; the end-to-end data flow; one page per detail flow.
- ``post.pdf`` (4:5 pages): what a LinkedIn post needs: the post text, ready to copy,
  then the diagrams as images to attach (or to upload together as a document carousel).

Layout is deterministic and lives in ``templates/*.typ``; the model only ever produces
data. Content is passed as JSON through Typst's ``sys.inputs``, so paper text cannot
inject markup. Neither PDF carries a creation date, so the same run renders the same bytes.
"""

from __future__ import annotations

import json
import struct
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

import typst

from labmate.core.theme import FONTS_DIR, Theme
from labmate.paper2flow.schemas import Flows, Paper, Post, WrittenSlides
from labmate.paper2flow.steps.flow import DPI, LETTERS, legend

CARD_COLORS = {
    "Task": "#fff8d5",
    "Challenges": "#ffe8e7",
    "Proposed method": "#e3f2f8",
    "Main results": "#e4f8d6",
}
"""Card backgrounds of the project pages on adamfodor.com."""

FLOW_KICKERS = {
    "method": "End-to-end data flow",
    "benchmark": "How the benchmark works",
    "survey": "How the survey maps the field",
    "position": "How the argument flows",
}
"""Heading of the overview diagram, per paper type."""


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


def blocks(slides: WrittenSlides, labels: list[str]) -> list[dict[str, Any]]:
    """The four blocks as cards (citations dropped).

    Args:
        slides: Fact-checked blocks.
        labels: One label per block ("Task", ...).

    Returns:
        ``{"label", "title", "bullets", "color"}`` per block.
    """
    return [
        {
            "label": label,
            "title": s.title,
            "bullets": [b.text for b in s.bullets],
            "color": CARD_COLORS.get(label, "#f1e1dc"),
        }
        for s, label in zip(slides.slides, labels, strict=True)
    ]


def natural_size(png: Path) -> tuple[float, float]:
    """Size in points of a Graphviz PNG at its render resolution (read from the header).

    A small diagram is shown at this size at most, so its boxes aren't blown up to fill
    the page.

    Args:
        png: PNG file rendered at :data:`~labmate.paper2flow.steps.flow.DPI`.

    Returns:
        ``(width, height)`` in points.
    """
    width, height = struct.unpack(">II", png.read_bytes()[16:24])
    return width * 72 / DPI, height * 72 / DPI


def diagrams(
    flows: Flows, images: list[str], paper_type: str, image_dir: Path
) -> list[dict[str, Any]]:
    """One page description per diagram: kicker, title, caption, image and legend.

    Args:
        flows: All diagrams.
        images: Their PNG file names (overview first).
        paper_type: Route (picks the overview's kicker).
        image_dir: Where the PNGs are (the output directory).

    Returns:
        Page data, overview first.
    """
    labels = {n.id: n.label for n in flows.overview.nodes}
    kickers = [FLOW_KICKERS.get(paper_type, FLOW_KICKERS["method"])] + [
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


def _compile(template: str, data: dict[str, Any], out: Path) -> Path:
    """Compile ``templates/<template>`` with ``data`` into ``out``.

    Typst resolves image paths relative to the source file, so the template is copied
    next to the output (where the images are) for the compile and removed afterwards.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    source = out.with_suffix(".typ")
    source.write_text(files("labmate.paper2flow.templates").joinpath(template).read_text())
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
        cards: The four blocks (:func:`blocks`).
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
    return _compile("overview.typ", data, out)


def post_text(post: Post, paper: Paper) -> list[str]:
    """The post as paragraphs, in the order they are pasted into LinkedIn.

    Args:
        post: Fact-checked post.
        paper: The paper (title and link).

    Returns:
        Hook, takeaways, question, then the paper reference.
    """
    reference = f"Paper: {paper.title}" + (f" {paper.url}" if paper.url else "")
    return [post.hook, *(t.text for t in post.takeaways), post.question, reference]


def render_post(
    post: Post,
    paper: Paper,
    pages: list[dict[str, Any]],
    out: Path,
    theme: Theme | None = None,
) -> Path:
    """Render ``post.pdf``: the post text, then one 4:5 image page per diagram.

    Args:
        post: Fact-checked post.
        paper: The paper.
        pages: The diagrams (:func:`diagrams`); the first one is headed by the hook.
        out: Output PDF path.
        theme: Colours and font.

    Returns:
        ``out``.
    """
    data = {
        "hook": post.hook,
        "takeaways": [t.text for t in post.takeaways],
        "question": post.question,
        "title": paper.title,
        "authors": author_line(paper.authors),
        "publication": publication_line(paper),
        "url": paper.url,
        "diagrams": pages,
        "theme": (theme or Theme()).model_dump(),
    }
    return _compile("post.typ", data, out)
