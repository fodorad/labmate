import pytest
from pydantic import BaseModel

from paper2carousel.llm.structured import (
    StructuredOutputError,
    extract_json,
    parse_structured,
    schema_instruction,
    structured_chat,
)
from paper2carousel.llm.types import ChatRequest, Message


class Claim(BaseModel):
    claim: str
    page: int


def test_strict_json_parses_strictly():
    obj, mode = parse_structured('{"claim": "x", "page": 3}', Claim)
    assert obj == Claim(claim="x", page=3) and mode == "strict"


def test_fenced_json_parses_leniently():
    text = 'Here you go:\n```json\n{"claim": "x", "page": 3}\n```\nHope it helps!'
    obj, mode = parse_structured(text, Claim)
    assert obj == Claim(claim="x", page=3) and mode == "lenient"


def test_json_embedded_in_prose_parses_leniently():
    obj, mode = parse_structured('Sure! {"claim": "a {b}", "page": 1} Done.', Claim)
    assert obj.claim == "a {b}" and mode == "lenient"


def test_wrong_schema_or_no_json_fails():
    assert parse_structured('{"claim": "x"}', Claim) == (None, None)
    assert parse_structured("no json at all", Claim) == (None, None)


def test_extract_json_edge_cases():
    assert extract_json("} backwards {") is None
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'


def test_schema_instruction_embeds_the_schema():
    text = schema_instruction(Claim)
    assert '"claim"' in text and '"page"' in text
    assert "only the JSON" in text


# --- structured_chat: schema prompt + retry-with-feedback loop -----------------------------


REQ = ChatRequest(model="qwen3.6:35b-mlx", messages=[Message(role="user", content="extract")])


def test_structured_chat_sends_schema_both_ways(fake):
    fake.chat_handler = lambda b: {
        "model": b["model"],
        "message": {"content": '{"claim":"x","page":2}'},
    }
    assert structured_chat(fake.client(), REQ, Claim) == Claim(claim="x", page=2)
    body = fake.requests[0][1]
    assert body["messages"][0]["role"] == "system" and '"page"' in body["messages"][0]["content"]
    assert body["format"]["properties"].keys() == {"claim", "page"}


def test_structured_chat_retries_with_validation_feedback(fake):
    replies = iter(['{"claim": "x"}', '```json\n{"claim":"x","page":2}\n```'])
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": next(replies)}}
    assert structured_chat(fake.client(), REQ, Claim).page == 2
    retry = fake.requests[1][1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": '{"claim": "x"}'}
    assert "page" in retry[-1]["content"] and "Field required" in retry[-1]["content"]


def test_structured_chat_gives_up_after_budget(fake):
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": "no json"}}
    with pytest.raises(StructuredOutputError, match="no valid Claim after 2 attempts"):
        structured_chat(fake.client(), REQ, Claim, max_retries=1)
    assert len(fake.requests) == 2
