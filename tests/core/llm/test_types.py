import base64

import pytest

from labmate.core.llm.types import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
    Message,
    Usage,
    canonical_json,
)


def _req(**kw):
    return ChatRequest(model="m", messages=[Message(role="user", content="hi")], **kw)


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == canonical_json(
        {"a": [1, {"c": 3, "d": 2}], "b": 1}
    )


def test_chat_payload_contains_only_set_fields():
    payload = _req(think=None, seed=None).to_payload()
    assert payload == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"temperature": 0.0},
    }


def test_chat_payload_full():
    schema = {"type": "object"}
    tools = [{"type": "function", "function": {"name": "f"}}]
    payload = _req(format=schema, tools=tools, num_ctx=8192, keep_alive=0, seed=7).to_payload()
    assert payload["format"] == schema
    assert payload["tools"] == tools
    assert payload["options"] == {"temperature": 0.0, "seed": 7, "num_ctx": 8192}
    assert payload["think"] is False
    assert payload["keep_alive"] == 0


def test_cache_key_ignores_keep_alive_and_schema_key_order():
    a = _req(format={"type": "object", "properties": {"x": {}, "y": {}}}, keep_alive="5m")
    b = _req(format={"properties": {"y": {}, "x": {}}, "type": "object"}, keep_alive=0)
    assert a.cache_key() == b.cache_key()


def test_cache_key_changes_with_content_seed_and_digest():
    base = _req()
    assert base.cache_key() != _req(seed=1).cache_key()
    assert (
        base.cache_key()
        != ChatRequest(model="m", messages=[Message(role="user", content="other")]).cache_key()
    )
    assert base.cache_key("sha256:aaa") != base.cache_key("sha256:bbb")
    assert base.cache_key("sha256:aaa") == base.cache_key("sha256:aaa")


def test_chat_and_image_keys_never_collide():
    assert _req().cache_key() != ImageRequest(model="m", prompt="hi").cache_key()


def test_usage_from_ollama_converts_ns_to_ms():
    usage = Usage.from_ollama(
        {
            "prompt_eval_count": 10,
            "eval_count": 40,
            "load_duration": 2_500_000_000,
            "eval_duration": 2_000_000_000,
            "total_duration": 5_000_000_000,
        }
    )
    assert usage.load_ms == 2500
    assert usage.tokens_per_s == 20
    assert usage.total_ms == 5000


def test_usage_tokens_per_s_zero_without_eval_time():
    assert Usage(completion_tokens=5).tokens_per_s == 0.0


def test_chat_response_parses_tool_calls_and_thinking():
    response = ChatResponse.from_ollama(
        {
            "model": "m",
            "message": {
                "content": "",
                "thinking": "hmm",
                "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
            },
        }
    )
    assert response.tool_calls[0].function.name == "f"
    assert response.tool_calls[0].function.arguments == {"a": 1}
    assert response.thinking == "hmm"
    assert response.cached is False


def test_chat_response_empty_thinking_becomes_none():
    assert ChatResponse.from_ollama({"message": {"content": "x", "thinking": ""}}).thinking is None


def test_image_request_payload():
    payload = ImageRequest(model="img", prompt="p", steps=8, seed=3, keep_alive=0).to_payload()
    assert payload == {
        "model": "img",
        "prompt": "p",
        "width": 1080,
        "height": 1350,
        "stream": False,
        "steps": 8,
        "options": {"seed": 3},
        "keep_alive": 0,
    }
    assert "options" not in ImageRequest(model="img", prompt="p", seed=None).to_payload()


def test_image_response_roundtrip():
    raw = b"\x89PNGfake"
    response = ImageResponse.from_ollama(
        {"model": "img", "image": base64.b64encode(raw).decode(), "total_duration": 1_000_000}
    )
    assert response.image_bytes() == raw
    assert response.total_ms == 1.0
    assert len(response.sha256()) == 64


def test_image_response_without_image_raises():
    with pytest.raises(ValueError, match="no 'image'"):
        ImageResponse.from_ollama({"model": "qwen", "response": "text"})


def test_num_predict_goes_into_options():
    assert _req(num_predict=64).to_payload()["options"]["num_predict"] == 64
    assert "num_predict" not in _req().to_payload()["options"]


def test_tool_call_arguments_have_a_canonical_key_order():
    from labmate.core.llm.types import ToolFunction

    # a live response and its cassette (written with sorted keys) must look the same
    live = ToolFunction(
        name="make_chart", arguments={"values": [1], "labels": ["a"], "x": {"b": 1, "a": 2}}
    )
    assert list(live.arguments) == ["labels", "values", "x"]
    assert list(live.arguments["x"]) == ["a", "b"]
