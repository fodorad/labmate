import numpy as np
import pytest

from labmate.ask.build import build_index, chapter_text
from labmate.ask.chunk import (
    MIN_SECTION_WORDS,
    THESIS_MAX_WORDS,
    DocSection,
    chunk_document,
    find_heading,
    outline_sections,
    section_number,
    sentences,
    thesis_chunks,
)
from labmate.ask.embed import Embedder
from labmate.ask.index import Index, fts_query, rrf
from labmate.ask.library import LibraryError, Source, load_library
from labmate.config import ReplayMode
from labmate.core.ingest import IngestError
from labmate.core.llm.replay import CassetteMissError, CassetteStore, ReplayClient
from labmate.core.llm.types import EmbedRequest
from labmate.core.tracing import TracedClient, Tracer, read_trace
from tests.ask.conftest import DISSERTATION, make_doc

# --- library ------------------------------------------------------------------------------


def test_library_manifest(library):
    lib = load_library(library)
    assert [s.id for s in lib.sources] == ["dissertation", "blinklinmult", "outside"]
    assert lib.get("blinklinmult").theses == ["I"]
    assert lib.path(lib.get("dissertation")) == library / "dissertation.pdf"
    assert Source(id="x", file="x.pdf").name == "x"
    with pytest.raises(KeyError):
        lib.get("nope")


@pytest.mark.parametrize(
    ("toml", "error"),
    [
        (None, "no .*library.toml"),
        ('[[source]]\nid = "a"\nfile = "a.pdf"\ntier = 1\n[[source]]\nid = "a"\nfile = "b.pdf"\n',
         "duplicate source ids"),
        ('[[source]]\nid = "a"\nfile = "a.pdf"\ntier = 2\n', "no tier-1 source"),
    ],
)  # fmt: skip
def test_library_errors(tmp_path, toml, error):
    if toml is not None:
        (tmp_path / "library.toml").write_text(toml)
    with pytest.raises(LibraryError, match=error):
        load_library(tmp_path)


# --- chunking -----------------------------------------------------------------------------


def test_section_tree_numbers_paths_and_references(tmp_path):
    sections = outline_sections(make_doc(tmp_path / "d.pdf", DISSERTATION), "d")
    assert [(s.number, s.title, s.page) for s in sections] == [
        ("1", "Introduction", 1), ("2", "Method", 2), ("2.1", "Architecture", 2),
        ("2.2", "Training", 3), ("3", "Theses", 4),
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


def test_chunks_captions_and_theses(tmp_path):
    sections = outline_sections(make_doc(tmp_path / "d.pdf", DISSERTATION), "diss")
    source = Source(id="diss", file="d.pdf", tier=1)
    chunks = chunk_document(sections, source, chunk_words=30)
    assert all(len(c.text.split()) <= 45 for c in chunks if c.kind == "text")
    assert chunks[0].id == "diss:0000" and len({c.id for c in chunks}) == len(chunks)
    caption = next(c for c in chunks if c.kind == "caption")
    assert caption.text.startswith("Figure 1: The BlinkLinMulT") and caption.page == 2
    theses = [c for c in chunks if c.kind == "thesis"]
    assert [c.thesis for c in theses] == ["I", "II"]
    assert theses[1].text.startswith("Thesis II: Training on a union")
    # papers (tier 2) have no thesis chunks
    paper = chunk_document(sections, source.model_copy(update={"tier": 2}), 30)
    assert not any(c.kind == "thesis" for c in paper)
    assert thesis_chunks([sections[0]], source, 0) == []


def test_theses_stated_in_the_introduction_are_found():
    long = " ".join(["word"] * 400)
    text = (
        "Some context.\nThesis 1. I propose LinMulT, a linear transformer.\n"
        f"Thesis 2. I estimate perceived personality in groups.\nThesis 3. {long}\n"
        "Thesis 3. repeated later with more than five words here."
    )
    intro = DocSection(id="d:s000", source_id="d", level=1, number="1", title="Introduction",
                       path=["1 Introduction"], page=1, text=text, breaks=[])  # fmt: skip
    other = intro.model_copy(update={"id": "d:s001", "title": "Method", "text": "Thesis 1. x"})
    chunks = thesis_chunks([other, intro], Source(id="d", file="", tier=1), 7)
    assert [c.thesis for c in chunks] == ["1", "2", "3"]
    assert chunks[0].id == "d:0007" and chunks[0].section_id == "d:s000"
    assert len(chunks[2].text.split()) == 2 + THESIS_MAX_WORDS


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
    chunks = chunk_document(sections, Source(id="l", file="", tier=2), 20)
    assert {c.page for c in chunks if c.section_id == "l:s000"} == {1, 2}


# --- embeddings and the backend -----------------------------------------------------------


def test_embed_request_key_client_replay_and_trace(tmp_path, fake):
    request = EmbedRequest(model="embeddinggemma:latest", input=["a text"], keep_alive="5m")
    assert request.cache_key() == request.model_copy(update={"keep_alive": None}).cache_key()
    assert request.cache_key("d1") != request.cache_key("d2")
    live = fake.client().embed(request)
    assert len(live.embeddings) == 1 and live.prompt_tokens == 7 and not live.cached
    store = CassetteStore(tmp_path / "c")
    ReplayClient(fake.client(), store, ReplayMode.RECORD).embed(request)
    replayed = ReplayClient(None, store, ReplayMode.REPLAY).embed(request)
    assert replayed.cached and replayed.embeddings == live.embeddings
    with pytest.raises(CassetteMissError):
        ReplayClient(None, store, ReplayMode.REPLAY).embed(
            request.model_copy(update={"input": ["x"]})
        )
    tracer = Tracer(tmp_path / "t.jsonl")
    TracedClient(fake.client(), tracer).embed(request)
    span = read_trace(tmp_path / "t.jsonl")[0]
    assert span["name"] == "llm.embed" and span["n_inputs"] == 1


def test_embedder_prefixes_batches_and_normalises(fake):
    embedder = Embedder(fake.client(), "embeddinggemma:latest", batch=2)
    vectors = embedder.documents(["blink detection", "eye blinks", "attention tokens"])
    assert vectors.shape == (3, 64) and np.allclose(np.linalg.norm(vectors, axis=1), 1)
    embeds = [b for p, b in fake.requests if p == "/api/embed"]
    assert len(embeds) == 2 and embeds[0]["input"][0].startswith("title: none | text: ")
    query = embedder.query("blink")
    assert fake.requests[-1][1]["input"] == ["task: search result | query: blink"]
    assert float(vectors[0] @ query) > float(vectors[2] @ query)
    assert Embedder(fake.client(), "other-model").query_prefix == ""


# --- index --------------------------------------------------------------------------------


def test_fts_query_and_rrf():
    assert fts_query("What is the F1 of BlinkLinMulT?") == '"f1" OR "blinklinmult"'
    assert fts_query("what is it?") == ""
    assert rrf([["a", "b"], ["b", "c"]], k=1)[0][0] == "b"


def test_build_index_and_search(indexed, library):
    index = indexed.index
    assert [s.id for s in index.sources()] == ["dissertation", "blinklinmult", "outside"]
    assert index.meta()["embed_model"] == "embeddinggemma:latest"
    kinds = {c.kind for c in index.chunks("dissertation")}
    assert kinds == {"text", "caption", "thesis", "summary"}
    summary = next(c for c in index.chunks("dissertation") if c.kind == "summary")
    assert summary.text.startswith("Summary of Introduction: The chapter Introduction")
    assert not any(c.kind == "summary" for c in index.chunks("outside"))
    assert index.claims() and all(c.id.split(":")[0] != "outside" for c in index.claims())
    # lexical, dense and hybrid search, filtered by tier
    hits = index.search("CEW ZJU datasets", indexed.embedder.query("CEW ZJU datasets"), [1], 3)
    assert hits and all(h.chunk.tier == 1 for h in hits)
    best = next(h for h in hits if "CEW" in h.chunk.text)
    assert best.bm25_rank == 1 and best.score > 0
    assert index.search("attention tokens", None, [3], 2, "bm25")[0].chunk.source_id == "outside"
    dense = index.search("x", indexed.embedder.query("blink transformer"), [1, 2], 2, "dense")
    assert len(dense) == 2 and dense[0].score >= dense[1].score
    assert index.search("the", None, [1], 3, "bm25") == []
    assert index.dense(indexed.embedder.query("x"), [], 3) == []
    with pytest.raises(ValueError, match="needs the query vector"):
        index.search("x", None, [1], 3, "hybrid")
    # small-to-big: a chunk with its neighbours; a caption alone
    text_ids = [c.id for c in index.chunks("dissertation") if c.kind == "text"]
    arch = [c.id for c in index.chunks("dissertation")
            if c.kind == "text" and c.section_id == "dissertation:s002"]  # fmt: skip
    assert [c.id for c in index.neighbors(arch[0])] == arch[:2]
    assert text_ids
    caption = next(c for c in index.chunks("dissertation") if c.kind == "caption")
    assert index.neighbors(caption.id) == [caption]
    assert index.section("dissertation:s002").path == ["Method", "Architecture"]
    for lookup in (index.chunk, index.section, index.source):
        with pytest.raises(KeyError):
            lookup("nope")


def test_build_reuses_and_refreshes(indexed, library, model):
    lib = load_library(library)
    calls = len(model.requests)
    assert build_index(indexed, lib) == {}  # everything indexed already
    assert len(model.requests) == calls
    done = build_index(indexed, lib, fresh=True, claims=False)
    assert set(done) == {"dissertation", "blinklinmult", "outside"}
    assert done["dissertation"][1] == 0 and indexed.index.claims() == []


def test_index_in_replay_mode_needs_no_model(indexed, config, library):
    from labmate.ask.session import open_ask

    config.ask.index = library / "again.sqlite"
    again = open_ask(config, ReplayMode.REPLAY)
    build_index(again, load_library(library))
    assert len(again.index.chunks()) == len(indexed.index.chunks())
    again.close()


def test_chapter_text_includes_subsections(tmp_path):
    sections = outline_sections(make_doc(tmp_path / "d.pdf", DISSERTATION), "d")
    text = chapter_text(sections, sections[1])
    assert "introduces" in text and "Heavy augmentation" in text and "Thesis I" not in text


def test_index_remove_source(tmp_path, indexed):
    index = Index(indexed.config.ask.index)
    assert index.has_source("outside")
    index.remove_source("outside")
    assert not index.has_source("outside") and index.bm25("attention tokens", [3], 3) == []
    index.close()


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
    chunks = chunk_document(sections, Source(id="t", file="", tier=2), 180)
    assert {c.section_id for c in chunks} == {"t:s001", "t:s002", "t:s003"}  # bare part: none
    assert len(sections[0].text.split()) >= MIN_SECTION_WORDS  # letter-spaced, still no chunk
