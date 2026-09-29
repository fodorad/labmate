from labmate.ask.agent import ask_prebuilt, build_agent, make_tools, parse_answer
from labmate.core.tracing import read_trace


def test_prebuilt_agent_answers_with_cited_sentences_only(indexed):
    answer = ask_prebuilt(indexed, build_agent(indexed), "CEW ZJU datasets")
    assert answer.agent == "prebuilt" and not answer.abstained
    assert len(answer.sentences) == 1  # the uncited sentence is left out
    assert answer.citations[0].chunk_id.startswith("dissertation:")
    root = [s for s in read_trace(indexed.tracer.path) if s["name"] == "run"][-1]
    assert root["agent"] == "prebuilt" and root["tool_calls"] == 1


def test_tools(indexed):
    search, read, sources = make_tools(indexed)
    out = search.invoke({"query": "CEW datasets", "scope": "own"})
    assert out.startswith("[") and "Dissertation" in out
    assert search.invoke({"query": "zzzz qqqq", "scope": "all"}).startswith("[")  # dense finds some
    first = indexed.index.chunks("dissertation")[0].id
    assert read.invoke({"chunk_id": first}).startswith(f"[{first}] Dissertation")
    assert "unknown chunk id" in read.invoke({"chunk_id": "nope:0000"})
    assert "dissertation: Dissertation (tier 1)" in sources.invoke({})


def test_parse_answer(indexed):
    first = indexed.index.chunks("dissertation")[0].id
    answer = parse_answer(indexed, "q", f"It works [{first}]. Made up [nope:0001]. No citation.")
    assert [s.chunk_ids for s in answer.sentences] == [[first]]
    empty = parse_answer(indexed, "q", "I could not find it.")
    assert empty.abstained and empty.text == "I could not find it."
