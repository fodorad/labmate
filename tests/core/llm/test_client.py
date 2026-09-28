import httpx
import pytest

from labmate.core.llm.client import Backend, OllamaClient, OllamaError, normalize_tag
from labmate.core.llm.types import ChatRequest, ImageRequest, Message


def test_normalize_tag():
    assert normalize_tag("x/flux2-klein") == "x/flux2-klein:latest"
    assert normalize_tag("qwen3.6:35b-mlx") == "qwen3.6:35b-mlx"
    assert normalize_tag("host:5000/org/model") == "host:5000/org/model:latest"


def test_client_satisfies_backend_protocol(fake):
    assert isinstance(fake.client(), Backend)


def test_chat_roundtrip(fake):
    with fake.client() as client:
        response = client.chat(
            ChatRequest(model="qwen3.6:35b-mlx", messages=[Message(role="user", content="hi")])
        )
    assert response.content == "OK seed=42"
    assert response.usage.completion_tokens == 50
    assert fake.requests[0][0] == "/api/chat"
    assert fake.requests[0][1]["stream"] is False


def test_generate_image(fake):
    response = fake.client().generate_image(ImageRequest(model="x/flux2-klein:latest", prompt="p"))
    assert response.image_bytes().startswith(b"\x89PNG")


def test_unload_and_running_models(fake):
    client = fake.client()
    client.chat(ChatRequest(model="gemma4:26b-mlx", messages=[Message(role="user")]))
    assert client.running_models() == ["gemma4:26b-mlx"]
    client.unload("gemma4:26b-mlx")
    assert client.running_models() == []


def test_list_show_version(fake):
    client = fake.client()
    assert client.list_models()["gemma4:26b-mlx"].startswith("21c59a2eae30")
    assert "vision" in client.show("gemma4:26b-mlx")["capabilities"]
    assert client.version() == "0.24.0"


def test_http_error_carries_server_message(fake):
    with pytest.raises(OllamaError, match="404: model 'nope' not found"):
        fake.client().chat(ChatRequest(model="nope", messages=[]))


def test_non_json_error_body():
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(OllamaError, match="500: boom"):
        OllamaClient(transport=transport).version()


def test_connection_error_is_wrapped():
    def refuse(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        OllamaClient(transport=httpx.MockTransport(refuse)).version()


def test_unload_can_wait_until_model_is_gone(fake):
    client = fake.client()
    client.chat(ChatRequest(model="gemma4:26b-mlx", messages=[Message(role="user")]))
    fake.unload_delay_polls = 3
    assert client.unload("gemma4:26b-mlx", wait_s=5, poll_s=0) is True
    assert client.running_models() == []


def test_unload_wait_times_out_for_stuck_model(fake):
    client = fake.client()
    client.chat(ChatRequest(model="gemma4:26b-mlx", messages=[Message(role="user")]))
    fake.sticky.add("gemma4:26b-mlx")
    assert client.unload("gemma4:26b-mlx", wait_s=0.05, poll_s=0.01) is False


def test_a_dropped_connection_is_retried_once():
    import httpx

    from labmate.core.llm.client import OllamaClient, OllamaError

    calls = []

    def flaky(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, json={"version": "0.24.0"})

    assert OllamaClient(transport=httpx.MockTransport(flaky)).version() == "0.24.0"

    def dead(request):
        raise httpx.RemoteProtocolError("Server disconnected")

    import pytest

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        OllamaClient(transport=httpx.MockTransport(dead)).version()
