from labmate.ask.build import build_index
from labmate.ask.library import load_library


def embeds(model):
    return [p for p, _ in model.requests if p == "/api/embed"]


def test_an_indexed_library_is_not_embedded_again(indexed, library, model):
    done = len(embeds(model))

    assert build_index(indexed, load_library(library)) == {}
    assert len(embeds(model)) == done


def test_fresh_reindexes_everything(indexed, library):
    done = build_index(indexed, load_library(library), fresh=True)

    assert set(done) == {"dissertation", "blinklinmult", "outside"}


def test_changed_index_settings_rebuild_every_source(indexed, library):
    indexed.config.ask.chunk_words = 60

    done = build_index(indexed, load_library(library))

    assert set(done) == {"dissertation", "blinklinmult", "outside"}
    assert indexed.index.meta()["chunk_words"] == "60"
    assert build_index(indexed, load_library(library)) == {}  # now up to date
