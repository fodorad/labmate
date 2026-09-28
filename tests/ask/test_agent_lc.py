import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from labmate.ask.agent import ask_prebuilt, build_agent, make_tools, parse_answer
from labmate.ask.lc import (
    LibraryRetriever,
    RecordedChatModel,
    RecordedEmbeddings,
    to_messages,
)
from labmate.core.tracing import read_trace


@tool
def lookup(term: str) -> str:
    """Look a term up.

    Args:
        term: The term.
    """
    return term


def test_messages_convert_both_ways():
    messages = to_messages([
        SystemMessage("sys"),
        HumanMessage("hi"),
        AIMessage("", tool_calls=[{"name": "lookup", "args": {"b": 1, "a": 2}, "id": "x"}]),
        ToolMessage("result", tool_call_id="x", name="lookup"),
    ])  # fmt: skip
    assert [m.role for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2].tool_calls[0].function.arguments == {"a": 2, "b": 1}
    assert messages[3].tool_name == "lookup"

    class Weird(HumanMessage):
        type: str = "weird"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="unsupported message type"):
        to_messages([Weird("x")])


def test_recorded_chat_model_goes_through_the_backend(session, model):
    chat = RecordedChatModel(llm=session.llm).bind_tools([lookup])
    reply = chat.invoke([HumanMessage("Which datasets?")])
    assert reply.tool_calls[0]["name"] == "search_library"
    assert reply.tool_calls[0]["id"] == "call_0_0"
    assert reply.usage_metadata["total_tokens"] == 0  # the fake reports no usage here
    sent = [b for p, b in model.requests if p == "/api/chat"][-1]
    assert sent["tools"][0]["function"]["name"] == "lookup"
    assert any(s["name"] == "llm.chat" for s in read_trace(session.tracer.path))


def test_embeddings_and_retriever(indexed):
    embeddings = RecordedEmbeddings(indexed.embedder)
    assert len(embeddings.embed_query("blink")) == 64
    assert len(embeddings.embed_documents(["a", "b"])) == 2
    retriever = LibraryRetriever(index=indexed.index, embedder=indexed.embedder, tiers=(1,), k=2)
    docs = retriever.invoke("CEW ZJU datasets")
    assert len(docs) == 2 and all(d.metadata["tier"] == 1 for d in docs)
    assert docs[0].id and "score" in docs[0].metadata
    lexical = LibraryRetriever(index=indexed.index, embedder=indexed.embedder, method="bm25")
    assert "CEW" in lexical.invoke("CEW ZJU")[0].page_content


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
