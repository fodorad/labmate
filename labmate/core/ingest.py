"""Ingest: fetch a paper and split it into sections. Plain code, no LLM.

Sections come from the PDF outline (most arXiv PDFs built with ``hyperref`` have one).
Without an outline the text is split per page. The references section is dropped.
"""

from __future__ import annotations

import hashlib
import html
import re
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx
import pymupdf
from pydantic import BaseModel

from labmate.core.figures import extract_figures
from labmate.core.schemas import Paper, Section

ARXIV_PDF = "https://arxiv.org/pdf/{id}"
"""PDF download URL template."""

ARXIV_API = "https://export.arxiv.org/api/query?id_list={id}"
"""Metadata (Atom feed) URL template."""

ARXIV_ABS = "https://arxiv.org/abs/{id}"
"""Abstract page URL template (metadata fallback)."""

USER_AGENT = "labmate (https://github.com/fodorad/labmate)"
"""Sent with every request; arXiv asks API clients to identify themselves."""

API_ATTEMPTS = 3
"""Tries per arXiv API request."""

RETRY_WAIT_S = 3.0
"""Base wait between tries (multiplied by the attempt number)."""

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
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


def _get_with_retry(http: httpx.Client, url: str, wait_s: float) -> httpx.Response:
    """GET with a few retries: the arXiv API answers 406/429/5xx now and then."""
    for attempt in range(API_ATTEMPTS):
        response = http.get(url)
        if response.status_code not in (406, 429) and response.status_code < 500:
            break
        if attempt + 1 < API_ATTEMPTS:
            time.sleep(wait_s * (attempt + 1))
    response.raise_for_status()
    return response


class ArxivMetadata(BaseModel):
    """Metadata from the arXiv API.

    Attributes:
        title: Title.
        authors: Author names.
        abstract: Abstract.
        published: Date of the first version (``YYYY-MM-DD``).
        notes: The journal reference and author comments (where a venue is often named).
    """

    title: str
    authors: list[str]
    abstract: str
    published: str = ""
    notes: str = ""


def fetch_metadata(
    arxiv_id: str, http: httpx.Client, retry_wait_s: float = RETRY_WAIT_S
) -> ArxivMetadata:
    """Fetch title, authors and abstract from the arXiv API.

    Args:
        arxiv_id: Bare arXiv id.
        http: HTTP client.
        retry_wait_s: Base wait between API tries (see :data:`RETRY_WAIT_S`).

    Returns:
        Title, authors and abstract.

    Raises:
        IngestError: If the API returns no entry for the id.
    """
    try:
        response = _get_with_retry(http, ARXIV_API.format(id=arxiv_id), retry_wait_s)
    except httpx.HTTPStatusError:
        # the export API sometimes refuses single ids (406) for good; the abstract page
        # carries the same metadata in its citation_* tags
        return _metadata_from_abs_page(arxiv_id, http)
    entry = ET.fromstring(response.text).find("a:entry", _ATOM)
    title = entry.findtext("a:title", "", _ATOM) if entry is not None else ""
    if entry is None or not title.strip() or title.strip() == "Error":
        raise IngestError(f"arXiv has no entry for {arxiv_id}")

    def clean(s: str) -> str:
        return " ".join(s.split())

    authors = entry.findall("a:author", _ATOM)
    notes = [
        f"{label}: {clean(text)}"
        for label, tag in (
            ("Journal reference", "arxiv:journal_ref"),
            ("Comments", "arxiv:comment"),
        )
        if (text := entry.findtext(tag, "", _ATOM)).strip()
    ]
    return ArxivMetadata(
        title=clean(title),
        authors=[clean(a.findtext("a:name", "", _ATOM)) for a in authors],
        abstract=clean(entry.findtext("a:summary", "", _ATOM)),
        published=entry.findtext("a:published", "", _ATOM).strip()[:10],
        notes="\n".join(notes),
    )


def _metadata_from_abs_page(arxiv_id: str, http: httpx.Client) -> ArxivMetadata:
    """Title, authors and abstract from the ``citation_*`` meta tags of the abstract page."""
    response = http.get(ARXIV_ABS.format(id=arxiv_id))
    response.raise_for_status()

    def meta(name: str) -> list[str]:
        pattern = rf'<meta\s+name="citation_{name}"\s+content="([^"]*)"'
        return [" ".join(html.unescape(v).split()) for v in re.findall(pattern, response.text)]

    titles = meta("title")
    if not titles:
        raise IngestError(f"arXiv has no entry for {arxiv_id}")
    authors = [" ".join(reversed(a.split(", ", 1))) for a in meta("author")]  # "Last, First"
    abstract = meta("abstract")
    dates = meta("date")  # "2017/06/12"
    return ArxivMetadata(
        title=titles[0],
        authors=authors,
        abstract=abstract[0] if abstract else "",
        published=dates[0].replace("/", "-") if dates else "",
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
        if not 1 <= page <= len(pages) or _REFERENCES.search(f"\n{title.strip()}\n"):
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
        date=meta.published,
        year=_year(meta.published),
        notes=meta.notes,
        sections=split_sections(pdf),
        figures=extract_figures(pdf, run_dir / "figures", run_dir),
    )


def _year(date: str) -> int | None:
    """Leading four-digit year of a date string (``2017-06-12``, ``20230921...``)."""
    match = re.match(r"(19|20)\d\d", date)
    return int(match.group(0)) if match else None


def first_page_text(pdf: Path, limit: int = 4000) -> str:
    """Text of the PDF's first page, where venue and publication date are usually printed.

    Args:
        pdf: PDF path.
        limit: Characters to keep.

    Returns:
        The text, whitespace-normalised per line.
    """
    with pymupdf.open(pdf) as doc:
        text = doc[0].get_text()
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)[:limit]


def is_url(ref: str) -> bool:
    """True for ``http(s)://`` references.

    Args:
        ref: Path or URL.

    Returns:
        Whether ``ref`` is a URL.
    """
    return ref.startswith(("http://", "https://"))


def url_stem(url: str) -> str:
    """File stem for a URL: its path's stem plus a short hash of the whole URL.

    The hash keeps papers apart whose URLs end in the same file name (every OpenReview
    link ends in ``/pdf``), so they never share a download or a run directory, e.g.
    ``2023_Fodor_Adam_MDPI_BlinkLinMulT-1a2b3c4d``.

    Args:
        url: PDF URL.

    Returns:
        The stem (the run id is its slug).
    """
    digest = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"{PurePosixPath(urlparse(url).path).stem or 'paper'}-{digest}"


def download_url(url: str, cache_dir: Path, http: httpx.Client) -> Path:
    """Download a PDF from a URL unless it is already cached.

    Args:
        url: PDF URL.
        cache_dir: Download cache (``runs/.downloads``).
        http: HTTP client.

    Returns:
        The local PDF, named after the URL (so the run id is stable).

    Raises:
        IngestError: If the response is not a PDF.
    """
    dest = cache_dir / f"{url_stem(url)}.pdf"
    if not dest.exists():
        response = http.get(url)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise IngestError(f"{url} did not return a PDF")
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
    return dest


def ingest_pdf(
    pdf: Path, title: str | None = None, url: str = "", run_dir: Path | None = None
) -> Paper:
    """Ingest a local PDF (for papers that are not on arXiv).

    Args:
        pdf: PDF path.
        title: Title override; defaults to the PDF metadata title or the file name.
        url: Link to show on the slides.
        run_dir: Where the PDF is cached (``paper.pdf``) and figure crops go
            (``run_dir/figures``); no figures if omitted.

    Returns:
        The ingested paper. Authors come from the PDF metadata; the abstract from a
        section titled "Abstract", else from the metadata subject (LaTeX/hyperref PDFs
        often carry it there).
    """
    if run_dir is not None and pdf.resolve() != (run_dir / "paper.pdf").resolve():
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, run_dir / "paper.pdf")
    with pymupdf.open(pdf) as doc:
        meta = doc.metadata or {}
    sections = split_sections(pdf)
    abstract = next((s.text for s in sections if s.title.lower() == "abstract"), "")
    authors = [a.strip() for a in re.split(r",| and ", meta.get("author") or "") if a.strip()]
    return Paper(
        paper_id=slugify(pdf.stem),
        title=title or meta.get("title") or pdf.stem,
        authors=authors,
        abstract=abstract or " ".join((meta.get("subject") or "").split()),
        url=url,
        # the PDF's creation date is usually close to publication: a fallback for the year
        year=_year((meta.get("creationDate") or "").removeprefix("D:")),
        sections=sections,
        figures=extract_figures(pdf, run_dir / "figures", run_dir) if run_dir else [],
    )
