import httpx
import pymupdf
import pytest

from labmate.core.ingest import (
    IngestError,
    download_pdf,
    download_url,
    fetch_metadata,
    first_page_text,
    ingest_arxiv,
    ingest_pdf,
    parse_arxiv_id,
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


def test_ingest_local_pdf_takes_authors_and_abstract_from_metadata(tmp_path):
    meta = {
        "author": "Ádám Fodor, Kristian Fenech and András Lőrincz",
        "subject": "We  detect blinks.",
    }
    paper = ingest_pdf(make_pdf(tmp_path / "blink.pdf", metadata=meta), run_dir=tmp_path / "run")
    assert paper.authors == ["Ádám Fodor", "Kristian Fenech", "András Lőrincz"]
    assert paper.abstract == "We detect blinks."
    assert (tmp_path / "run" / "paper.pdf").exists()  # cached next to the artifacts


def test_download_url_caches_and_rejects_non_pdfs(tmp_path):
    pdf_bytes = make_pdf(tmp_path / "src.pdf").read_bytes()
    hits = []

    def handle(request):
        hits.append(request.url.path)
        if request.url.path.endswith("html.pdf"):
            return httpx.Response(200, content=b"<html>login</html>")
        return httpx.Response(200, content=pdf_bytes)

    http = httpx.Client(transport=httpx.MockTransport(handle))
    out = download_url("https://x.org/a/My_Paper.pdf", tmp_path / "dl", http)
    assert out == tmp_path / "dl" / "My_Paper.pdf" and out.read_bytes() == pdf_bytes
    download_url("https://x.org/a/My_Paper.pdf", tmp_path / "dl", http)
    assert len(hits) == 1
    with pytest.raises(IngestError, match="did not return a PDF"):
        download_url("https://x.org/html.pdf", tmp_path / "dl", http)


def test_metadata_falls_back_to_the_abstract_page():
    page = (
        '<meta name="citation_title" content="Survey on Evaluation of LLM-based Agents" />'
        '<meta name="citation_author" content="Yehudai, Asaf" />'
        '<meta name="citation_author" content="Eden, Lilach" />'
        '<meta name="citation_abstract" content="We survey  agents &amp; benchmarks." />'
        '<meta name="citation_date" content="2025/03/20" />'
    )

    def handle(request):
        if "export.arxiv.org" in str(request.url):
            return httpx.Response(406)
        if "/abs/2503.16416" in str(request.url):
            return httpx.Response(200, text=page)
        return httpx.Response(200, text="<html></html>")

    http = httpx.Client(transport=httpx.MockTransport(handle))
    meta = fetch_metadata("2503.16416", http, retry_wait_s=0.0)
    assert meta.title == "Survey on Evaluation of LLM-based Agents"
    assert meta.authors == ["Asaf Yehudai", "Lilach Eden"]
    assert meta.abstract == "We survey agents & benchmarks."
    assert meta.published == "2025-03-20"
    with pytest.raises(IngestError, match="no entry"):
        fetch_metadata("2503.99999", http, retry_wait_s=0.0)


DATED_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>A Test Paper</title>
    <summary>Abstract.</summary>
    <published>2017-06-12T17:57:34Z</published>
    <author><name>Ada Lovelace</name></author>
    <arxiv:comment>15 pages, 5 figures</arxiv:comment>
    <arxiv:journal_ref>NeurIPS 30 (2017)</arxiv:journal_ref>
  </entry>
</feed>"""


def test_arxiv_publication_date_and_notes(tmp_path):
    fake = FakeArxiv(make_pdf(tmp_path / "src.pdf").read_bytes(), atom=DATED_ATOM)
    paper = ingest_arxiv("2401.00001", tmp_path / "run", fake.client())
    assert paper.date == "2017-06-12" and paper.year == 2017
    assert paper.notes == ("Journal reference: NeurIPS 30 (2017)\nComments: 15 pages, 5 figures")


def test_pdf_year_falls_back_to_the_creation_date_and_first_page_text(tmp_path):
    pdf = make_pdf(tmp_path / "p.pdf", metadata={"creationDate": "D:20230921120000Z"})
    assert ingest_pdf(pdf).year == 2023
    text = first_page_text(pdf, limit=30)
    assert text.startswith("1 Introduction") and len(text) == 30
