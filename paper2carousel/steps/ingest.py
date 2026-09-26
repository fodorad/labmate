"""Step 0: fetch a paper and split it into sections. Plain code, no LLM.

Sections come from the PDF outline (most arXiv PDFs built with ``hyperref`` have one).
Without an outline the text is split per page. The references section is dropped.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pymupdf
from pydantic import BaseModel

from paper2carousel.schemas import Paper, Section
from paper2carousel.steps.figures import extract_figures

ARXIV_PDF = "https://arxiv.org/pdf/{id}"
"""PDF download URL template."""

ARXIV_API = "https://export.arxiv.org/api/query?id_list={id}"
"""Metadata (Atom feed) URL template."""

_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_REFERENCES = re.compile(r"\n\s*(References|Bibliography|REFERENCES)\s*\n")


class IngestError(RuntimeError):
    """Raised when a paper cannot be fetched or parsed."""


def parse_arxiv_id(ref: str) -> str:
    """Extract a bare arXiv id from an id, ``arXiv:`` reference or URL.

    Args:
        ref: e.g. ``"1706.03762"``, ``"arXiv:1706.03762v7"`` or
            ``"https://arxiv.org/abs/1706.03762"``.

    Returns:
        The id without version, e.g. ``"1706.03762"``.

    Raises:
        IngestError: If no new-style arXiv id is found.
    """
    match = _ARXIV_ID.search(ref)
    if not match:
        raise IngestError(f"Not an arXiv id: {ref!r}")
    return match.group(1)


class ArxivMetadata(BaseModel):
    """Metadata from the arXiv API."""

    title: str
    authors: list[str]
    abstract: str


def fetch_metadata(arxiv_id: str, http: httpx.Client) -> ArxivMetadata:
    """Fetch title, authors and abstract from the arXiv API.

    Args:
        arxiv_id: Bare arXiv id.
        http: HTTP client.

    Returns:
        Title, authors and abstract.

    Raises:
        IngestError: If the API returns no entry for the id.
    """
    response = http.get(ARXIV_API.format(id=arxiv_id))
    response.raise_for_status()
    entry = ET.fromstring(response.text).find("a:entry", _ATOM)
    title = entry.findtext("a:title", "", _ATOM) if entry is not None else ""
    if entry is None or not title.strip() or title.strip() == "Error":
        raise IngestError(f"arXiv has no entry for {arxiv_id}")

    def clean(s: str) -> str:
        return " ".join(s.split())

    authors = entry.findall("a:author", _ATOM)
    return ArxivMetadata(
        title=clean(title),
        authors=[clean(a.findtext("a:name", "", _ATOM)) for a in authors],
        abstract=clean(entry.findtext("a:summary", "", _ATOM)),
    )


def download_pdf(arxiv_id: str, dest: Path, http: httpx.Client) -> Path:
    """Download the paper PDF unless it is already cached at ``dest``.

    Args:
        arxiv_id: Bare arXiv id.
        dest: Target file.
        http: HTTP client.

    Returns:
        ``dest``.
    """
    if not dest.exists():
        response = http.get(ARXIV_PDF.format(id=arxiv_id))
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
    return dest


def slugify(text: str) -> str:
    """Lower-case, filesystem-safe identifier.

    Args:
        text: Any string, e.g. a file stem.

    Returns:
        ``text`` with runs of non-alphanumerics replaced by ``-``.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _find_heading(text: str, title: str, start: int) -> int:
    """Offset of a heading at or after ``start``, tolerating numbering like "3.2 "."""
    pattern = re.compile(
        r"^\s*(\d+(\.\d+)*\.?\s+)?" + re.escape(title.strip()), re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(text, start)
    return match.start() if match else -1


def split_sections(pdf: Path, max_level: int = 1) -> list[Section]:
    """Split a PDF into sections using its outline, falling back to one section per page.

    Args:
        pdf: PDF path.
        max_level: Deepest outline level that starts a new section.

    Returns:
        Sections in reading order, with the references removed.

    Raises:
        IngestError: If the PDF has no extractable text.
    """
    with pymupdf.open(pdf) as doc:
        pages = [page.get_text() for page in doc]
        toc = [(title, page) for level, title, page in doc.get_toc() if level <= max_level]
    if not any(p.strip() for p in pages):
        raise IngestError(f"No extractable text in {pdf} (scanned PDF?)")

    text = ""
    page_offsets = []
    for page_text in pages:
        page_offsets.append(len(text))
        text += page_text + "\n"
    refs = _REFERENCES.search(text)
    end = refs.start() if refs else len(text)

    starts: list[tuple[str, int, int]] = []  # (title, page, offset)
    for title, page in toc:
        if not 1 <= page <= len(pages):
            continue
        offset = _find_heading(text, title, page_offsets[page - 1])
        if 0 <= offset < end and (not starts or offset > starts[-1][2]):
            starts.append((title.strip(), page, offset))

    if not starts:
        return [
            Section(title=f"Page {i + 1}", page=i + 1, text=t.strip())
            for i, t in enumerate(pages)
            if t.strip() and page_offsets[i] < end
        ]
    sections = []
    for i, (title, page, offset) in enumerate(starts):
        stop = starts[i + 1][2] if i + 1 < len(starts) else end
        sections.append(Section(title=title, page=page, text=text[offset:stop].strip()))
    return sections


def ingest_arxiv(ref: str, run_dir: Path, http: httpx.Client) -> Paper:
    """Fetch an arXiv paper (metadata + PDF) and split it into sections.

    Args:
        ref: arXiv id, reference or URL.
        run_dir: Run directory; the PDF is cached as ``paper.pdf`` there.
        http: HTTP client.

    Returns:
        The ingested paper.
    """
    arxiv_id = parse_arxiv_id(ref)
    meta = fetch_metadata(arxiv_id, http)
    pdf = download_pdf(arxiv_id, run_dir / "paper.pdf", http)
    return Paper(
        paper_id=arxiv_id,
        title=meta.title,
        authors=meta.authors,
        abstract=meta.abstract,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        sections=split_sections(pdf),
        figures=extract_figures(pdf, run_dir / "figures", run_dir),
    )


def ingest_pdf(
    pdf: Path, title: str | None = None, url: str = "", run_dir: Path | None = None
) -> Paper:
    """Ingest a local PDF (for papers that are not on arXiv).

    Args:
        pdf: PDF path.
        title: Title override; defaults to the PDF metadata title or the file name.
        url: Link to show on the slides.
        run_dir: Where figure crops go (``run_dir/figures``); no figures if omitted.

    Returns:
        The ingested paper. The abstract is taken from a section titled "Abstract" if any.
    """
    with pymupdf.open(pdf) as doc:
        meta_title = (doc.metadata or {}).get("title") or ""
    sections = split_sections(pdf)
    abstract = next((s.text for s in sections if s.title.lower() == "abstract"), "")
    return Paper(
        paper_id=slugify(pdf.stem),
        title=title or meta_title or pdf.stem,
        abstract=abstract,
        url=url,
        sections=sections,
        figures=extract_figures(pdf, run_dir / "figures", run_dir) if run_dir else [],
    )
