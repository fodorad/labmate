import httpx

from labmate.config import CacheConfig, Config
from labmate.core.chat import chat_model
from labmate.core.usage import UsageCollector, ollama_resident


def test_the_collector_sums_the_calls_and_counts_the_models_that_had_to_be_loaded_again(fake):
    gb = iter([21.0, 37.0, 16.0])  # what Ollama reports as loaded after each call
    collector = UsageCollector(resident=lambda: next(gb))
    config = Config(cache=CacheConfig(enabled=False), callbacks=[collector])
    transport = httpx.MockTransport(fake.handle)
    writer = chat_model(config, "qwen3.8:27b-mlx", transport=transport)
    judge = chat_model(config, "gemma4:26b-mlx", transport=transport)

    writer.invoke("a")
    judge.invoke("b")
    writer.invoke("c")  # the writer was pushed out by the judge and loads again

    usage = collector.summary()
    assert usage.calls == 3 and usage.models == ("qwen3.8:27b-mlx", "gemma4:26b-mlx")
    assert usage.tokens_in == 60 and usage.tokens_out == 150
    assert usage.loads == 3 and usage.swaps == 1 and usage.load_s == 6.0
    assert usage.peak_gb == 37.0  # both did not fit at the same time on a 32 GB Mac
    assert usage.decode_tps == 50.0  # 150 tokens in 3 s


def test_the_resident_memory_is_what_ollama_says_is_loaded():
    def ps(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        sizes = [{"name": "a", "size": 21_000_000_000}, {"name": "b", "size": 6_600_000_000}]
        return httpx.Response(200, json={"models": sizes})

    resident = ollama_resident("http://ollama", httpx.MockTransport(ps))

    assert resident() == 27.6
    assert (
        ollama_resident("http://ollama", httpx.MockTransport(lambda r: httpx.Response(500)))()
        == 0.0
    )
