import pytest

from labmate.ask.index import Index, fts_query, rrf


def test_fts_query_and_rrf():
    assert fts_query("What is the F1 of BlinkLinMulT?") == '"f1" OR "blinklinmult"'
    assert fts_query("what is it?") == ""
    assert rrf([["a", "b"], ["b", "c"]], k=1)[0][0] == "b"


def test_every_source_is_indexed_with_text_and_caption_chunks(indexed):
    index = indexed.index

    assert [s.id for s in index.sources()] == ["blinklinmult", "dissertation", "outside"]
    assert index.meta()["embed_model"] == "embeddinggemma:latest"
    assert {c.kind for c in index.chunks("dissertation")} == {"text", "caption"}


def test_hybrid_search_finds_the_chunk_with_the_words_first(indexed):
    hits = indexed.index.search("CEW ZJU datasets", indexed.embedder.query("CEW ZJU datasets"), 3)

    best = next(h for h in hits if "CEW" in h.chunk.text)
    assert best.bm25_rank == 1 and best.score > 0
    assert hits[0].score >= hits[-1].score


def test_dense_search_ranks_by_cosine(indexed):
    dense = indexed.index.dense(indexed.embedder.query("blink transformer"), 2)

    assert len(dense) == 2 and dense[0].score >= dense[1].score


def test_a_hit_is_read_with_its_neighbours_and_a_caption_alone(indexed):
    index = indexed.index
    arch = [c.id for c in index.chunks("dissertation")
            if c.kind == "text" and c.section_id == "dissertation:s002"]  # fmt: skip
    assert [c.id for c in index.neighbors(arch[0])] == arch[:2]
    caption = next(c for c in index.chunks("dissertation") if c.kind == "caption")
    assert index.neighbors(caption.id) == [caption]
    assert index.section("dissertation:s002").path == ["Method", "Architecture"]
    for lookup in (index.chunk, index.section, index.source):
        with pytest.raises(KeyError):
            lookup("nope")


def test_removing_a_source_removes_its_chunks_from_search(indexed):
    index = Index(indexed.config.ask.index_file)
    assert index.bm25("quadratically", 3)  # only the outside source says it

    index.remove_source("outside")

    assert not index.has_source("outside") and index.bm25("quadratically", 3) == []
    index.close()
