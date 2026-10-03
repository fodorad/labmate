import pytest

from labmate.ask.chunk import (
    MIN_SECTION_WORDS,
    chunk_document,
    find_heading,
    numeric_share,
    outline_sections,
    section_number,
    sentences,
)
from labmate.ask.library import Source
from labmate.core.ingest import IngestError
from tests.ask.conftest import DISSERTATION, make_doc


def test_section_tree_numbers_paths_and_references(tmp_path):
    sections = outline_sections(make_doc(tmp_path / "d.pdf", DISSERTATION), "d")
    assert [(s.number, s.title, s.page) for s in sections] == [
        ("1", "Introduction", 1), ("2", "Method", 2), ("2.1", "Architecture", 2),
        ("2.2", "Training", 3),
    ]  # fmt: skip
    assert sections[3].path == ["Method", "Training"] and sections[3].level == 2
    assert all("A cited paper" not in s.text for s in sections)  # references dropped
    assert section_number(sections[2]) == "§2.1"
    assert section_number(sections[2].model_copy(update={"number": ""})) == "Architecture"


def test_pages_without_outline_and_scanned_pdfs(tmp_path):
    import pymupdf

    doc = pymupdf.open()
    for text in ["First page text here.", "Second page text here."]:
        doc.new_page().insert_text((72, 72), text)
    doc.save(tmp_path / "plain.pdf")
    sections = outline_sections(tmp_path / "plain.pdf", "p")
    assert [(s.title, s.page) for s in sections] == [("Page 1", 1), ("Page 2", 2)]
    blank = pymupdf.open()
    blank.new_page()
    blank.save(tmp_path / "blank.pdf")
    with pytest.raises(IngestError):
        outline_sections(tmp_path / "blank.pdf", "b")


def test_sentences_keep_decimals_together():
    text = "It reaches 0.912 F1. Then, e.g. the next one. (A note.) Last"
    assert [s for _, s in sentences(text)] == [
        "It reaches 0.912 F1.", "Then, e.g. the next one.", "(A note.)", "Last",
    ]  # fmt: skip


def test_text_is_chunked_by_sentences_and_captions_get_their_own_chunk(tmp_path):
    sections = outline_sections(make_doc(tmp_path / "d.pdf", DISSERTATION), "diss")
    source = Source(id="diss", file="d.pdf")

    chunks = chunk_document(sections, source, chunk_words=30)

    assert all(len(c.text.split()) <= 45 for c in chunks if c.kind == "text")
    assert chunks[0].id == "diss:0000" and len({c.id for c in chunks}) == len(chunks)
    caption = next(c for c in chunks if c.kind == "caption")
    assert caption.text.startswith("Figure 1: The BlinkLinMulT") and caption.page == 2
    assert {c.kind for c in chunks} == {"text", "caption"}


def test_pages_follow_page_breaks_inside_a_section(tmp_path):
    body = " ".join(f"Sentence number {i} of the long section." for i in range(40))
    pdf = make_doc(tmp_path / "long.pdf", [(1, "1 Long", 1, body[:900]), (1, "2 End", 3, "End.")])
    import pymupdf

    doc = pymupdf.open(pdf)  # add a second page of the same section
    page = doc.new_page(pno=1)
    page.insert_textbox(pymupdf.Rect(72, 72, 520, 600), body[900:], fontsize=9)
    doc.set_toc([[1, "Long", 1], [1, "End", 3]])
    doc.save(tmp_path / "long2.pdf")
    sections = outline_sections(tmp_path / "long2.pdf", "l")
    assert sections[0].breaks
    chunks = chunk_document(sections, Source(id="l", file=""), 20)
    assert {c.page for c in chunks if c.section_id == "l:s000"} == {1, 2}


def test_find_heading_tolerates_letter_spacing_and_numbers():
    text = (
        "intro\n1\nI N T R O D U C T I O N\nbody\n2.1\nHuman Factors\n"
        "P S Y C H O L O G I C A L A N D\nF O U N D AT I O N S\n"
    )
    assert find_heading(text, "1 Introduction", 0) == text.index("1\nI N T")
    assert find_heading(text, "2.1 Human Factors", 0) == text.index("2.1")
    assert find_heading(text, "Psychological and Foundations", 0) == text.index("P S Y")
    assert find_heading(text, "Missing", 0) == -1
    assert find_heading(text, "  ", 0) == -1


def test_thesis_layout_parts_front_matter_and_listed_bibliography(tmp_path):
    import pymupdf

    doc = pymupdf.open()
    pages = [
        "C O N T E N T S\n1 Introduction 3\nBibliography 5",
        "P A R T I\nI N T R O D U C T I O N",
        "1\nI N T R O D U C T I O N\n" + "The thesis studies blinks in long videos. " * 5
        + "\n1.1\nM O T I V A T I O N\n" + "Blinks tell a lot about attention. " * 5,
        "2\nR E S U L T S\n" + "The model reaches 0.912 F1 on the benchmark. " * 5,
        "B I B L I O G R A P H Y\n[1] Someone. A paper. 2020.",
    ]  # fmt: skip
    for body in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 560, 800), body, fontsize=9)
    doc.set_toc([
        [1, "Contents", 1], [1, " Introduction", 2], [2, "1 Introduction", 3],
        [3, "1.1 Motivation", 3], [2, "2 Results", 4], [1, " Bibliography", 5],
    ])  # fmt: skip
    doc.save(tmp_path / "thesis.pdf")
    sections = outline_sections(tmp_path / "thesis.pdf", "t")
    assert [(s.level, s.number, s.title) for s in sections] == [
        (1, "", "Introduction"), (1, "1", "Introduction"), (2, "1.1", "Motivation"),
        (1, "2", "Results"),
    ]  # fmt: skip
    assert "Someone" not in sections[-1].text and "0.912" in sections[-1].text
    chunks = chunk_document(sections, Source(id="t", file=""), 180)
    assert {c.section_id for c in chunks} == {"t:s001", "t:s002", "t:s003"}  # bare part: none
    assert len(sections[0].text.split()) >= MIN_SECTION_WORDS  # letter-spaced, still no chunk


def test_flattened_tables_are_not_chunked_but_their_caption_is(tmp_path):
    table = "N dmodel dff h dk dv base 6 512 2048 8 64 64 0.1 0.1 100K 4.92 25.8 65 " * 3
    prose = "The model reaches strong results on the benchmark after careful tuning of it."
    caption = "Table 3: Variations on the Transformer architecture."
    sections = outline_sections(
        make_doc(tmp_path / "t.pdf", [(1, "1 Results", 1, f"{prose}\n{table}\n{caption}")]), "t"
    )

    chunks = chunk_document(sections, Source(id="t", file=""), chunk_words=20)

    assert numeric_share(table) > 0.4 > numeric_share(prose)
    assert any(c.kind == "caption" and c.text.startswith("Table 3") for c in chunks)
    assert all(numeric_share(c.text) <= 0.4 for c in chunks if c.kind == "text")
