import httpx
import pymupdf
import pytest

from paper2carousel.steps.ingest import (
    IngestError,
    download_pdf,
    fetch_metadata,
    ingest_arxiv,
    ingest_pdf,
    parse_arxiv_id,
    slugify,
    split_sections,
)
from tests.conftest import EMPTY_ATOM, FakeArxiv, make_pdf


@pytest.mark.parametrize(
    "ref",
    [
        "1706.03762",
        "arXiv:1706.03762v7",
        "https://arxiv.org/abs/1706.03762",
        "arxiv.org/pdf/1706.03762v2",
    ],
)
def test_parse_arxiv_id(ref):
    assert parse_arxiv_id(ref) == "1706.03762"


def test_parse_arxiv_id_rejects_garbage():
    with pytest.raises(IngestError):
        parse_arxiv_id("not a paper")


def test_sections_follow_outline_and_drop_references(tmp_path):
    sections = split_sections(make_pdf(tmp_path / "p.pdf"))
    assert [s.title for s in sections] == ["Introduction", "Method", "Results"]
    assert [s.page for s in sections] == [1, 2, 3]
    assert "84.6%" in sections[2].text
    assert all("A cited paper" not in s.text for s in sections)


def test_without_outline_falls_back_to_pages(tmp_path):
    sections = split_sections(make_pdf(tmp_path / "p.pdf", toc=False))
    assert [s.title for s in sections] == ["Page 1", "Page 2", "Page 3"]


def test_outline_entries_that_cannot_be_located_are_skipped(tmp_path):
    path = make_pdf(tmp_path / "p.pdf")
    with pymupdf.open(path) as doc:
        doc.set_toc([[1, "Introduction", 1], [1, "Nonexistent Heading", 2], [1, "Results", 99]])
        doc.saveIncr()
    assert [s.title for s in split_sections(path)] == ["Introduction"]


def test_pdf_without_text_raises(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    doc.save(tmp_path / "blank.pdf")
    with pytest.raises(IngestError, match="No extractable text"):
        split_sections(tmp_path / "blank.pdf")


def test_fetch_metadata_normalises_whitespace(arxiv):
    meta = fetch_metadata("2401.00001", arxiv.client())
    assert meta.title == "A Test Paper"
    assert meta.authors == ["Ada Lovelace", "Alan Turing"]
    assert meta.abstract == "We study attention. It is linear now."


def test_fetch_metadata_unknown_id(tmp_path):
    fake = FakeArxiv(b"", atom=EMPTY_ATOM)
    with pytest.raises(IngestError, match="no entry"):
        fetch_metadata("2401.00001", fake.client())


def test_download_is_cached(arxiv, tmp_path):
    dest = tmp_path / "run" / "paper.pdf"
    download_pdf("2401.00001", dest, arxiv.client())
    download_pdf("2401.00001", dest, arxiv.client())
    assert arxiv.pdf_downloads == 1 and dest.read_bytes().startswith(b"%PDF")


def test_download_http_error_propagates(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        download_pdf("2401.00001", tmp_path / "p.pdf", client)


def test_ingest_arxiv_end_to_end(arxiv, tmp_path):
    paper = ingest_arxiv("https://arxiv.org/abs/2401.00001v2", tmp_path, arxiv.client())
    assert paper.paper_id == "2401.00001"
    assert paper.url == "https://arxiv.org/abs/2401.00001"
    assert paper.title == "A Test Paper" and len(paper.sections) == 3


def test_ingest_local_pdf_uses_abstract_section_and_slug(tmp_path):
    path = make_pdf(
        tmp_path / "My Paper (final).pdf",
        sections=[("Abstract", "We do things."), ("Introduction", "Context here.")],
    )
    paper = ingest_pdf(path, url="https://example.org")
    assert paper.paper_id == "my-paper-final"
    assert paper.title == "A Test Paper"  # from PDF metadata
    assert paper.abstract.endswith("We do things.")
    assert ingest_pdf(path, title="Override").title == "Override"


def test_slugify():
    assert slugify("  LinMulT: v2 (Final) ") == "linmult-v2-final"
