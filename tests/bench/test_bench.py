import json

import httpx

from labmate.bench.micro import ModelResult, judge_test, unload_all
from labmate.bench.report import micro_markdown, profile_rows
from labmate.config import Config, ProfileConfig
from tests.conftest import chat, reply


def verdicts(kind: str):
    """A judge that answers every bullet of a card with the same verdict."""

    def handler(body):
        bullets = body["messages"][-1]["content"].count("Bullet ")
        items = [{"bullet": n, "verdict": kind, "reason": "x"} for n in range(1, bullets + 1)]
        return reply(body, json.dumps({"verdicts": items}))

    return handler


def test_a_judge_that_trusts_everything_misses_every_mistake(fake):
    fake.chat_handler = verdicts("supported")

    trusting = judge_test(chat(fake, "gemma4:26b-mlx"))

    assert (trusting.caught, trusting.mistakes) == (0, 7)
    assert trusting.false_alarms == 0 and "number on the wrong thing" in trusting.missed


def test_a_judge_that_doubts_everything_catches_all_mistakes_and_raises_false_alarms(fake):
    fake.chat_handler = verdicts("unsupported")

    doubting = judge_test(chat(fake, "gemma4:26b-mlx"))

    assert (doubting.caught, doubting.false_alarms, doubting.clean) == (7, 5, 5)


def test_the_tables_say_which_profiles_fit_in_memory_together():
    config = Config(
        profiles={
            "current": ProfileConfig(text="big", critic="medium"),
            "mixed": ProfileConfig(text="medium", critic="small"),
            "single": ProfileConfig(text="medium", critic="medium"),
        }
    )
    sizes = {"big": 21.0, "medium": 16.0, "small": 6.6}

    rows = {row.profile: row for row in profile_rows(config, sizes, ram_gb=32)}

    assert not rows["current"].fits and rows["current"].gb == 37.0  # swaps on a 32 GB Mac
    assert rows["mixed"].fits and rows["mixed"].gb == 22.6
    assert rows["single"].gb == 16.0  # one model counts once
    results = [ModelResult(tag, size_gb=gb, load_s=4, decode_tps=30) for tag, gb in sizes.items()]
    text = micro_markdown(results, config, 32)
    assert "**no, models are swapped**" in text and "`small`" in text


def test_unloading_asks_ollama_to_drop_every_loaded_model():
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "a"}, {"name": "b"}]})
        asked.append(request.content)
        return httpx.Response(200, json={})

    unload_all("http://ollama", httpx.MockTransport(handler))

    assert len(asked) == 2 and all(b'"keep_alive":0' in a.replace(b" ", b"") for a in asked)
