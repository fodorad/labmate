"""Crop figures out of a paper PDF (part of ingest). Plain code, no LLM.

Many arXiv figures are vector drawings rather than embedded images, so extracting
images alone misses them. Instead, for every caption starting with "Figure N", the
graphics (images and vector drawings) between the previous caption on the page and this
caption are unioned into one box, and that region of the page is rendered to PNG.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from paper2carousel.schemas import Figure

_CAPTION = re.compile(r"^\s*(?:Figure|Fig\.)\s*(\d+)\s*[:.|]")

MIN_SIDE = 40.0
"""Crops smaller than this (points) on either side are ignored."""

PADDING = 4.0
"""Margin added around the graphics box (points)."""

DPI = 200
"""Render resolution for the PNG crops."""

MAX_CAPTION = 300
"""Captions are truncated to this many characters."""


def _graphics(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]
    for img in page.get_images(full=True):
        rects.extend(page.get_image_rects(img[0]))
    return [r for r in rects if r.width > 2 or r.height > 2]


LABEL_GAP = 24.0
"""Text blocks this close above or beside the graphics box count as figure labels."""


def _label_floor(inside: list[pymupdf.Rect], floor: float) -> float:
    """Upper limit for label search: the previous caption, never below the graphics."""
    return min(floor, min(r.y0 for r in inside))


def _with_labels(box: pymupdf.Rect, blocks: list, floor: float) -> pymupdf.Rect:
    """Grow ``box`` to include short text blocks touching it (sub-figure titles, axis labels).

    Captions and long paragraphs are excluded, so body text never leaks into a crop.
    """
    grown = pymupdf.Rect(box)
    for x0, y0, x1, y1, text, *_ in blocks:
        r = pymupdf.Rect(x0, y0, x1, y1)
        near = (r.y1 <= box.y0 and box.y0 - r.y1 <= LABEL_GAP) or r.intersects(box)
        overlaps = r.x1 > box.x0 and r.x0 < box.x1
        if near and overlaps and r.y0 >= floor and len(text) < 80 and not _CAPTION.match(text):
            grown |= r
    return grown


def extract_figures(pdf: Path, out_dir: Path, rel_to: Path) -> list[Figure]:
    """Find captioned figures and save each as a PNG crop.

    Args:
        pdf: Paper PDF.
        out_dir: Directory for the PNGs (created if needed).
        rel_to: Directory the stored paths are relative to (the paper's run directory).

    Returns:
        Figures in order of first appearance, one per figure number.
    """
    figures: list[Figure] = []
    seen: set[int] = set()
    with pymupdf.open(pdf) as doc:
        for page in doc:
            graphics = _graphics(page)
            floor = page.rect.y0
            blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
            for _x0, y0, _x1, y1, text, *_ in blocks:
                match = _CAPTION.match(text)
                if not match:
                    continue
                number = int(match.group(1))
                inside = [r for r in graphics if r.y0 >= floor - 1 and r.y1 <= y0 + 2]
                floor_before, floor = floor, y1
                if number in seen or not inside:
                    continue
                box = pymupdf.Rect(inside[0])
                for r in inside[1:]:
                    box |= r
                box = _with_labels(box, blocks, floor=_label_floor(inside, floor_before))
                box = (box + (-PADDING, -PADDING, PADDING, PADDING)) & page.rect
                if box.width < MIN_SIDE or box.height < MIN_SIDE:
                    continue
                seen.add(number)
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"fig{number}.png"
                page.get_pixmap(clip=box, dpi=DPI).save(path)
                caption = " ".join(text.split())
                figures.append(
                    Figure(
                        id=f"fig{number}",
                        number=number,
                        caption=caption[:MAX_CAPTION],
                        page=page.number + 1,
                        path=str(path.relative_to(rel_to)),
                    )
                )
    return figures
