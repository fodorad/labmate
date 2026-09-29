from labmate.ask.build import build_index, chapter_text
from labmate.ask.chunk import (
    outline_sections,
)
from labmate.ask.library import load_library
from labmate.config import ReplayMode
from tests.ask.conftest import DISSERTATION, make_doc


def test_build_reuses_and_refreshes(indexed, library, model):
    lib = load_library(library)
    calls = len(model.requests)
    assert build_index(indexed, lib) == {}  # everything indexed already
    assert len(model.requests) == calls
    done = build_index(indexed, lib, fresh=True, claims=False)
    assert set(done) == {"dissertation", "blinklinmult", "outside"}
    assert done["dissertation"][1] == 0 and indexed.index.claims() == []


def test_changed_index_settings_rebuild_every_source(indexed, library):
    indexed.config.ask.chunk_words = 60
    done = build_index(indexed, load_library(library), claims=False)
    assert set(done) == {"dissertation", "blinklinmult", "outside"}
    assert indexed.index.meta()["chunk_words"] == "60"
    assert build_index(indexed, load_library(library), claims=False) == {}  # now up to date


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
