import pytest

from labmate.ask.index import Index, fts_query, rrf


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


def test_index_remove_source(tmp_path, indexed):
    index = Index(indexed.config.ask.index)
    assert index.has_source("outside")
    index.remove_source("outside")
    assert not index.has_source("outside") and index.bm25("attention tokens", [3], 3) == []
    index.close()
