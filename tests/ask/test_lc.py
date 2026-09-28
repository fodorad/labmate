import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

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
