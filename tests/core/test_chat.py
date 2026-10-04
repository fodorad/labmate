import httpx
from langchain_core.messages import HumanMessage

from labmate.config import CacheConfig, Config
from labmate.core.chat import SQLiteCache, chat_model
from tests.conftest import FakeOllama


def make_config(tmp_path, enabled: bool = True) -> Config:
    return Config(cache=CacheConfig(enabled=enabled, path=tmp_path / "cache" / "replies.sqlite"))


def chat_calls(fake: FakeOllama) -> int:
    return fake.paths().count("/api/chat")


def test_a_repeated_request_is_answered_from_the_cache(fake, tmp_path):
    model = chat_model(make_config(tmp_path), "qwen3.8:27b-mlx",
                       transport=httpx.MockTransport(fake.handle))  # fmt: skip

    first = model.invoke("hello").content
    second = model.invoke("hello").content

    assert first == second == "OK seed=42"
    assert chat_calls(fake) == 1


def test_the_cache_survives_a_restart(fake, tmp_path):
    config = make_config(tmp_path)
    transport = httpx.MockTransport(fake.handle)
    for _ in range(2):  # the second round is a fresh process opening the same file
        cache = SQLiteCache(config.cache.path)
        chat_model(config, "qwen3.8:27b-mlx", cache=cache, transport=transport).invoke("hello")
        cache.close()

    assert chat_calls(fake) == 1


def test_a_different_prompt_or_model_is_a_new_request(fake, tmp_path):
    config = make_config(tmp_path)
    transport = httpx.MockTransport(fake.handle)
    writer = chat_model(config, "qwen3.8:27b-mlx", transport=transport)
    judge = chat_model(config, "gemma4:26b-mlx", transport=transport)

    writer.invoke("hello")
    writer.invoke("goodbye")
    judge.invoke("hello")

    assert chat_calls(fake) == 3


def test_with_the_cache_off_every_request_reaches_the_model(fake, tmp_path):
    model = chat_model(make_config(tmp_path, enabled=False), "qwen3.8:27b-mlx",
                       transport=httpx.MockTransport(fake.handle))  # fmt: skip

    model.invoke("hello")
    model.invoke("hello")

    assert chat_calls(fake) == 2


def test_a_different_output_schema_is_a_new_request(fake, tmp_path):
    config = make_config(tmp_path)
    transport = httpx.MockTransport(fake.handle)

    chat_model(config, "qwen3.8:27b-mlx", transport=transport,
               json_schema={"type": "object"}).invoke("hello")  # fmt: skip
    chat_model(config, "qwen3.8:27b-mlx", transport=transport,
               json_schema={"type": "array"}).invoke("hello")  # fmt: skip

    assert chat_calls(fake) == 2


def test_a_cached_reply_keeps_its_tool_calls(fake, tmp_path):
    tool = {"type": "function", "function": {"name": "use_figure", "parameters": {}}}
    model = chat_model(make_config(tmp_path), "qwen3.8:27b-mlx",
                       transport=httpx.MockTransport(fake.handle))  # fmt: skip
    model = model.bind_tools([tool])
    messages = [HumanMessage("which figure?")]

    live = model.invoke(messages)
    cached = model.invoke(messages)

    assert chat_calls(fake) == 1
    assert cached.tool_calls[0]["name"] == live.tool_calls[0]["name"] == "use_paper_figure"
    assert cached.tool_calls[0]["args"] == {"figure_id": "fig3"}
