import pytest
from pydantic import BaseModel

from labmate.core.lc import batch_map, prompt, structured
from labmate.core.structured import StructuredOutputError
from tests.conftest import chat


class Claim(BaseModel):
    claim: str
    page: int


def replies(fake, *contents):
    answers = iter(contents)
    fake.chat_handler = lambda b: {
        "model": b["model"],
        "message": {"role": "assistant", "content": next(answers)},
    }


def chats(fake):
    return [body for path, body in fake.requests if path == "/api/chat"]


def test_structured_sends_the_schema_both_ways(fake):
    replies(fake, '{"claim": "x", "page": 2}')
    chain = prompt("Extract a claim from {text}") | structured(chat(fake), Claim)
    assert chain.invoke({"text": "the paper"}) == Claim(claim="x", page=2)
    body = chats(fake)[0]
    assert body["messages"][0]["role"] == "system" and '"page"' in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "Extract a claim from the paper"
    assert body["format"]["properties"].keys() == {"claim", "page"}


def test_structured_accepts_json_wrapped_in_prose_and_fences(fake):
    replies(fake, 'Sure!\n```json\n{"claim": "x", "page": 2}\n```')
    assert (prompt("Extract") | structured(chat(fake), Claim)).invoke({}).page == 2


def test_structured_feeds_schema_errors_and_broken_rules_back(fake):
    replies(fake, '{"claim": "x"}', '{"claim": "x", "page": 0}', '{"claim": "x", "page": 3}')
    rules = structured(chat(fake), Claim, check=lambda c: ["page 0 does not exist"] * (c.page < 1))
    assert (prompt("Extract") | rules).invoke({}).page == 3
    second, third = chats(fake)[1]["messages"], chats(fake)[2]["messages"]
    assert second[-2] == {"role": "assistant", "content": '{"claim": "x"}'}
    assert "Field required" in second[-1]["content"]
    assert "page 0 does not exist" in third[-1]["content"]


def test_structured_gives_up_after_its_budget(fake):
    replies(fake, "no json", "still none")
    rules = structured(chat(fake), Claim, max_retries=1)
    with pytest.raises(StructuredOutputError, match="no valid Claim after 2 attempts"):
        (prompt("Extract") | rules).invoke({})
    assert len(chats(fake)) == 2


def test_batch_map_keeps_the_input_order():
    assert batch_map(lambda x, cfg: x * 2, list(range(20)), None, workers=4) == [
        x * 2 for x in range(20)
    ]
