import pytest

from labmate.config import ReplayMode
from labmate.core.llm.replay import CassetteStore, ReplayClient
from labmate.core.llm.types import ChatRequest, ImageRequest, Message
from labmate.core.tracing import TracedClient, Tracer, read_trace


def test_nested_spans_link_to_parent_and_finish_inner_first():
    tracer = Tracer()
    with tracer.span("step", step="2") as outer:
        with tracer.span("llm.chat"):
            pass
        outer["claims"] = 5
    inner, outer_span = tracer.spans
    assert inner["name"] == "llm.chat"
    assert inner["parent_id"] == outer_span["span_id"]
    assert outer_span["parent_id"] is None
    assert outer_span["claims"] == 5 and outer_span["step"] == "2"
    assert {s["trace_id"] for s in tracer.spans} == {tracer.trace_id}


def test_error_is_recorded_and_reraised():
    tracer = Tracer()
    with pytest.raises(KeyError):
        with tracer.span("boom"):
            raise KeyError("x")
    assert tracer.spans[0]["status"] == "error"
    assert "KeyError" in tracer.spans[0]["error"]


def test_parent_is_restored_after_span():
    tracer = Tracer()
    with tracer.span("a"):
        pass
    with tracer.span("b"):
        pass
    assert all(s["parent_id"] is None for s in tracer.spans)


def test_jsonl_file_is_appended_and_readable(tmp_path):
    path = tmp_path / "run" / "trace.jsonl"
    tracer = Tracer(path, trace_id="r1")
    with tracer.span("a", n=1):
        pass
    with tracer.span("b"):
        pass
    spans = read_trace(path)
    assert [s["name"] for s in spans] == ["a", "b"]
    assert spans[0]["trace_id"] == "r1" and spans[0]["n"] == 1
    assert spans[0]["latency_ms"] >= 0


def test_traced_client_records_usage_and_cache_hits(fake, tmp_path):
    tracer = Tracer()
    client = TracedClient(
        ReplayClient(fake.client(), CassetteStore(tmp_path), ReplayMode.AUTO), tracer
    )
    request = ChatRequest(model="qwen3.6:35b-mlx", messages=[Message(role="user", content="x")])
    client.chat(request)
    client.chat(request)
    client.generate_image(ImageRequest(model="x/flux2-klein:latest", prompt="p"))
    first, second, image = tracer.spans
    assert first["cached"] is False and second["cached"] is True
    assert first["tokens_out"] == 50 and first["model"] == "qwen3.6:35b-mlx"
    assert first["key"] == request.cache_key()
    assert image["name"] == "llm.image" and len(image["image_sha256"]) == 64


def test_traced_client_keys_match_cassettes_when_digests_are_pinned(fake, tmp_path):
    from labmate.config import ReplayMode
    from labmate.core.llm.replay import CassetteStore, ReplayClient
    from labmate.core.llm.types import ChatRequest, Message

    digests = {"qwen3.6:35b-mlx": "1b50c6fdc2d4" + "0" * 52}
    store = CassetteStore(tmp_path / "c")
    tracer = Tracer()
    client = TracedClient(
        ReplayClient(fake.client(), store, ReplayMode.RECORD, digests), tracer, digests
    )
    request = ChatRequest(model="qwen3.6:35b-mlx", messages=[Message(role="user", content="hi")])
    client.chat(request)
    key = tracer.spans[0]["key"]
    assert key != request.cache_key() and store.path(key).exists()
