import httpx
import pytest

from labmate.config import CacheConfig, Config, OllamaConfig
from labmate.core.chat import chat_model
from labmate.core.ollama import DecisionError, OllamaServer


@pytest.fixture
def server(fake):
    return OllamaServer.from_config(Config(), fake.transport())


def test_unloading_drops_every_model_a_call_loaded(fake, server):
    chat_model(
        Config(cache=CacheConfig(enabled=False)), "gemma4:e4b", transport=fake.transport()
    ).invoke("hi")
    server.systemone("clef-flash", "a paper", {"relevance": {"type": "score"}})

    assert server.loaded() == {"clef-flash": 2.0, "gemma4:e4b": 6.6}
    assert server.resident_gb() == 8.6
    server.unload_all()
    assert server.loaded() == {} and fake.loaded == set()


def test_sizes_list_every_installed_model_also_without_latest(server):
    sizes = server.sizes()

    assert sizes["gemma4:26b-mlx"] == 16.0
    assert sizes["embeddinggemma"] == sizes["embeddinggemma:latest"] == 0.6


def test_capabilities_tell_a_decision_model_from_a_chat_model(server):
    assert server.capabilities("clef-flash") == ["decision"]
    assert "tools" in server.capabilities("gemma4:e4b")
    with pytest.raises(DecisionError, match="ollama pull nope"):
        server.capabilities("nope")


def test_a_decision_model_answers_and_a_chat_model_is_refused(fake, server):
    answers = server.systemone("clef-flash", "a paper", {"relevance": {"type": "score"}})

    assert answers["relevance"]["score"] == 2.0
    with pytest.raises(DecisionError, match="400"):
        server.systemone("gemma4:e4b", "a paper", {"relevance": {"type": "score"}})
    fake.systemone_handler = lambda body: {"answers": {}}
    with pytest.raises(DecisionError, match="no answers"):
        server.systemone("clef-flash", "a paper", {"relevance": {"type": "score"}})


def test_an_unreachable_server_reports_nothing_loaded_and_an_unknown_version(server):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down = OllamaServer("http://ollama", httpx.MockTransport(refuse))

    assert server.version() == "0.30.0"
    assert down.version() == "unknown" and down.resident_gb() == 0.0


def test_the_configured_host_is_used():
    config = Config(ollama=OllamaConfig(host="http://192.168.0.102:11434"))

    assert OllamaServer.from_config(config).host == "http://192.168.0.102:11434"
