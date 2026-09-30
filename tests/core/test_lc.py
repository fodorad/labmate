import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from labmate.config import ReplayMode
from labmate.core.lc import RecordedChatModel, batch_map, prompt, structured, to_messages
from labmate.core.llm.replay import CassetteStore, ReplayClient
from labmate.core.llm.structured import StructuredOutputError
from labmate.core.model import LLM


class Claim(BaseModel):
    claim: str
    page: int


@tool
def lookup(term: str) -> str:
    """Look a term up.

    Args:
        term: The term.
    """
    return term


def chat_model(fake, tmp_path, mode=ReplayMode.AUTO):
    backend = ReplayClient(fake.client(), CassetteStore(tmp_path / "cassettes"), mode)
    return RecordedChatModel(llm=LLM(backend, "qwen3.6:35b-mlx"))


def replies(fake, *contents):
    answers = iter(contents)
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": next(answers)}}


def chats(fake):
    return [body for path, body in fake.requests if path == "/api/chat"]


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


def test_every_reply_points_at_the_cassette_that_reproduces_it(fake, tmp_path):
    model = chat_model(fake, tmp_path)
    first = model.invoke([HumanMessage("hi")])
    again = model.invoke([HumanMessage("hi")])
    assert first.response_metadata["cached"] is False and again.response_metadata["cached"]
    key = first.response_metadata["cache_key"]
    assert key == again.response_metadata["cache_key"]
    assert CassetteStore(tmp_path / "cassettes").path(key).exists()
    assert len(chats(fake)) == 1


def test_tool_calls_get_deterministic_ids(fake, tmp_path):
    call = {"function": {"name": "lookup", "arguments": {"term": "blink"}}}
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"tool_calls": [call]}}
    reply = chat_model(fake, tmp_path).bind_tools([lookup]).invoke([HumanMessage("Look up?")])
    assert reply.tool_calls[0]["name"] == "lookup" and reply.tool_calls[0]["id"] == "call_0_0"
    assert chats(fake)[0]["tools"][0]["function"]["name"] == "lookup"


def test_structured_sends_the_schema_both_ways(fake, tmp_path):
    replies(fake, '{"claim": "x", "page": 2}')
    chain = prompt("Extract a claim from {text}") | structured(chat_model(fake, tmp_path), Claim)
    assert chain.invoke({"text": "the paper"}) == Claim(claim="x", page=2)
    body = chats(fake)[0]
    assert body["messages"][0]["role"] == "system" and '"page"' in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "Extract a claim from the paper"
    assert body["format"]["properties"].keys() == {"claim", "page"}


def test_structured_feeds_schema_errors_and_broken_rules_back(fake, tmp_path):
    replies(fake, '{"claim": "x"}', '{"claim": "x", "page": 0}', '{"claim": "x", "page": 3}')
    rules = structured(
        chat_model(fake, tmp_path), Claim, check=lambda c: ["page 0 does not exist"] * (c.page < 1)
    )
    assert (prompt("Extract") | rules).invoke({}).page == 3
    second, third = chats(fake)[1]["messages"], chats(fake)[2]["messages"]
    assert second[-2] == {"role": "assistant", "content": '{"claim": "x"}'}
    assert "Field required" in second[-1]["content"]
    assert "page 0 does not exist" in third[-1]["content"]


def test_structured_gives_up_after_its_budget(fake, tmp_path):
    replies(fake, "no json", "still none")
    rules = structured(chat_model(fake, tmp_path), Claim, max_retries=1)
    with pytest.raises(StructuredOutputError, match="no valid Claim after 2 attempts"):
        (prompt("Extract") | rules).invoke({})
    assert len(chats(fake)) == 2


def test_batch_map_keeps_the_input_order():
    assert batch_map(lambda x, cfg: x * 2, list(range(20)), None, workers=4) == [
        x * 2 for x in range(20)
    ]
