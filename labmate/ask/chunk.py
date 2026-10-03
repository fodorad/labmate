"""Split library documents into a section tree and retrieval-sized chunks. Plain code.

- **Sections** follow the PDF outline down to ``max_level`` (chapter › section ›
  subsection), with their numbering ("4.2") and path, so a hit can be cited as
  "Dissertation §4.2, p. 57" and expanded to its whole section (parent-document
  retrieval).
- **Chunks** pack whole sentences up to about ``chunk_words`` words, never across a
  section boundary, each with the page it starts on.
- **Tables** flattened into rows of numbers are not chunked (see :data:`MAX_NUMERIC_SHARE`).
- **Captions** ("Figure 3: ...", "Table 2 ...") become their own chunks: they are dense,
  and questions often ask about what a figure or table shows.
"""

from __future__ import annotations

import bisect
import re
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel

from labmate.ask.library import Source
from labmate.core.ingest import _REFERENCES, IngestError

ChunkKind = Literal["text", "caption"]
"""What a chunk holds: running text, or a figure or table caption."""

_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+")
_BOUNDARY = re.compile(r"([.!?][)\]\"']?)\s+(?=[A-Z0-9(\[\"'])")
"""A sentence end: punctuation, whitespace, then something that can start a sentence
(so "0.912" and "e.g. the" are not split)."""
_CAPTION = re.compile(
    r"(?m)^\s*((?:Figure|Fig\.|Table)\s*\d+[.:]?\s.{10,}?\.)(?=[ \t]*\n|\Z)", re.DOTALL
)
"""A caption: "Figure 3: ..." at a line start, up to the first sentence that ends a line."""

_NUMERIC = re.compile(r"^[\d.,%±×()\-+/:]+$")

MAX_NUMERIC_SHARE = 0.4
"""Text chunks with more numeric tokens than this are flattened tables and are left out: their
column names are far from the numbers, so a model reads the numbers wrongly (a layer count as
a head count). The table's caption is still indexed."""

MAX_CAPTION_WORDS = 80
"""Captions are cut to this length (long ones run into the body text)."""


class DocSection(BaseModel):
    """A node of a document's section tree.

    Attributes:
        id: ``<source>:s<nnn>``.
        source_id: Owning source.
        level: Outline level (1 = chapter).
        number: Heading number as printed ("4.2"), or empty.
        title: Heading without the number.
        path: Titles from the chapter down to this section.
        page: 1-based page where it starts.
        text: The section's own text (up to the next heading of any level).
        breaks: Offsets into ``text`` where a new page starts.
    """

    id: str
    source_id: str
    level: int
    number: str = ""
    title: str
    path: list[str]
    page: int
    text: str
    breaks: list[int] = []


class Chunk(BaseModel):
    """A retrievable piece of text.

    Attributes:
        id: ``<source>:<nnnn>``, stable for a given document and settings.
        source_id: Owning source.
        section_id: Section the chunk belongs to.
        kind: text or caption.
        page: 1-based page where the chunk starts.
        text: The chunk text (whitespace-normalised).
    """

    id: str
    source_id: str
    section_id: str
    kind: ChunkKind = "text"
    page: int
    text: str


def _clean(text: str) -> str:
    return " ".join(text.split())


_FRONT_MATTER = re.compile(
    r"^(table of )?contents$|^list of \w+$|^(kivonat|zusammenfassung|résumé)$", re.IGNORECASE
)
"""Outline entries that are not content: contents, lists of figures, tables, acronyms, and
the abstract in a second language."""


def find_heading(text: str, title: str, start: int) -> int:
    """Offset of a heading line at or after ``start``.

    Tolerates the section number being on its own line or missing, and letter-spaced
    headings ("I N T R O D U C T I O N") as LaTeX thesis classes set them.

    Args:
        text: Document text.
        title: Outline title, with or without its number ("2.1 Transformers").
        start: Where to start looking.

    Returns:
        The offset, or -1 if the heading is not found.
    """
    words = _NUMBER.sub("", title.strip()).split()
    if not words:
        return -1
    body = r"\s+".join(r"\s?".join(re.escape(ch) for ch in word) for word in words)
    pattern = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+)?" + body, re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text, start)
    return match.start() if match else -1


def outline_sections(pdf: Path, source_id: str, max_level: int = 3) -> list[DocSection]:
    """The document's section tree, from its PDF outline (one section per page without one).

    Args:
        pdf: PDF path.
        source_id: Source id (prefix of the section ids).
        max_level: Deepest outline level that starts a section.

    Returns:
        Sections in reading order, the references removed.

    Raises:
        IngestError: If the PDF has no extractable text.
    """
    with pymupdf.open(pdf) as doc:
        pages = [page.get_text() for page in doc]
        outline = [(lvl, t.strip(), p) for lvl, t, p in doc.get_toc()]
    if not any(p.strip() for p in pages):
        raise IngestError(f"No extractable text in {pdf} (scanned PDF?)")
    text, offsets = "", []
    for page_text in pages:
        offsets.append(len(text))
        text += page_text + "\n"
    # the bibliography: its outline entry if there is one, else the last "References"
    # heading (a thesis names it in its contents too)
    listed = [p for _, t, p in outline if _REFERENCES.search(f"\n{t}\n") and 1 <= p <= len(pages)]
    refs = list(_REFERENCES.finditer(text))
    if listed:
        start = offsets[listed[-1] - 1]
        found = [
            o for t in ("References", "Bibliography") if (o := find_heading(text, t, start)) >= 0
        ]
        end = min(found) if found else start
    else:
        end = refs[-1].start() if refs else len(text)
    # books and theses group chapters into unnumbered parts: count levels from the chapters
    shift = int(
        not any(_NUMBER.match(t) for lvl, t, _ in outline if lvl == 1)
        and any(_NUMBER.match(t) for lvl, t, _ in outline if lvl == 2)
    )
    toc = [(lvl - shift or 1, t, p) for lvl, t, p in outline if lvl <= max_level + shift]

    starts: list[tuple[int, str, int, int]] = []  # level, title, page, offset
    for level, title, page in toc:
        if (
            not 1 <= page <= len(pages)
            or _REFERENCES.search(f"\n{title}\n")
            or _FRONT_MATTER.match(title)
        ):
            continue
        offset = find_heading(text, title, offsets[page - 1])
        if 0 <= offset < end and (not starts or offset > starts[-1][3]):
            starts.append((level, title.strip(), page, offset))
    if not starts:
        starts = [(1, f"Page {i + 1}", i + 1, offsets[i]) for i in range(len(pages))]
        starts = [s for s in starts if s[3] < end]

    sections: list[DocSection] = []
    stack: list[str] = []
    for i, (level, title, page, offset) in enumerate(starts):
        stop = starts[i + 1][3] if i + 1 < len(starts) else end
        heading = _NUMBER.match(title) or _NUMBER.match(text[offset : offset + 40])
        number = heading.group(1) if heading else ""
        clean_title = _NUMBER.sub("", title).strip() or title
        stack = [*stack[: level - 1], clean_title]
        raw = text[offset:stop]
        lead = len(raw) - len(raw.lstrip())
        breaks = [o - offset - lead for o in offsets if offset < o < stop]
        sections.append(
            DocSection(
                id=f"{source_id}:s{i:03d}",
                source_id=source_id,
                level=level,
                number=number,
                title=clean_title,
                path=list(stack),
                page=page,
                text=raw.strip(),
                breaks=[b for b in breaks if b > 0],
            )
        )
    return sections


def page_at(section: DocSection, local: int) -> int:
    """Page of a character offset inside a section's text.

    Args:
        section: The section.
        local: Offset into ``section.text``.

    Returns:
        1-based page number.
    """
    return section.page + bisect.bisect_right(section.breaks, local)


def sentences(text: str) -> list[tuple[int, str]]:
    """Split text into sentences, with their start offsets.

    Args:
        text: Section text.

    Returns:
        ``(offset, sentence)`` pairs, empty sentences dropped.
    """
    marks = list(_BOUNDARY.finditer(text))
    starts = [0, *(m.end() for m in marks)]
    ends = [*(m.end(1) for m in marks), len(text)]
    return [(a, text[a:b].strip()) for a, b in zip(starts, ends, strict=True) if text[a:b].strip()]


def numeric_share(text: str) -> float:
    """The share of whitespace-separated tokens that are numbers.

    Args:
        text: Chunk text.

    Returns:
        A share from 0 to 1 (0 for empty text).
    """
    words = text.split()
    return sum(bool(_NUMERIC.match(w)) for w in words) / len(words) if words else 0.0


def chunk_section(
    section: DocSection, source: Source, chunk_words: int, start_index: int
) -> list[Chunk]:
    """Pack a section's sentences into chunks of about ``chunk_words`` words.

    Args:
        section: The section.
        source: Its source.
        chunk_words: Target size.
        start_index: Number of the first chunk (ids are numbered per source).

    Returns:
        Text chunks, then one chunk per caption found in the section.
    """
    chunks: list[Chunk] = []
    current: list[str] = []
    words, start = 0, 0
    index = start_index

    def flush() -> None:
        nonlocal current, words, index
        text = _clean(" ".join(current))
        if text and numeric_share(text) <= MAX_NUMERIC_SHARE:
            chunks.append(
                Chunk(
                    id=f"{source.id}:{index:04d}",
                    source_id=source.id,
                    section_id=section.id,
                    page=page_at(section, start),
                    text=text,
                )
            )
            index += 1
        current, words = [], 0

    for offset, sentence in sentences(section.text):
        n = len(sentence.split())
        if current and words + n > chunk_words:
            flush()
        if not current:
            start = offset
        current.append(sentence)
        words += n
    flush()
    for match in _CAPTION.finditer(section.text):
        caption = " ".join(_clean(match.group(1)).split()[:MAX_CAPTION_WORDS])
        chunks.append(
            Chunk(
                id=f"{source.id}:{index:04d}",
                source_id=source.id,
                section_id=section.id,
                kind="caption",
                page=page_at(section, match.start()),
                text=caption,
            )
        )
        index += 1
    return chunks


MIN_SECTION_WORDS = 12
"""Sections shorter than this (a heading followed by its first subsection) get no chunk."""


def chunk_document(
    sections: list[DocSection], source: Source, chunk_words: int = 180
) -> list[Chunk]:
    """All chunks of a document: section text and captions.

    Args:
        sections: The document's sections.
        source: The document.
        chunk_words: Target chunk size in words.

    Returns:
        Chunks with ids numbered in reading order.
    """
    chunks: list[Chunk] = []
    for section in sections:
        words = [w for w in section.text.split() if len(w) > 1]  # letter-spaced titles: none
        if len(words) >= MIN_SECTION_WORDS:  # not a bare heading
            chunks += chunk_section(section, source, chunk_words, len(chunks))
    return chunks


def section_number(section: DocSection) -> str:
    """How a section is cited: ``§4.2`` if numbered, else its title.

    Args:
        section: A section.

    Returns:
        The citation fragment.
    """
    return f"§{section.number}" if section.number else section.title
