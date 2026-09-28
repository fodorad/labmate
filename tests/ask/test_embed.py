import numpy as np
import pytest

from labmate.ask.embed import Embedder
from labmate.config import ReplayMode
from labmate.core.llm.replay import CassetteMissError, CassetteStore, ReplayClient
from labmate.core.llm.types import EmbedRequest
from labmate.core.tracing import TracedClient, Tracer, read_trace


def test_embed_request_key_client_replay_and_trace(tmp_path, fake):
    request = EmbedRequest(model="embeddinggemma:latest", input=["a text"], keep_alive="5m")
    assert request.cache_key() == request.model_copy(update={"keep_alive": None}).cache_key()
    assert request.cache_key("d1") != request.cache_key("d2")
    live = fake.client().embed(request)
    assert len(live.embeddings) == 1 and live.prompt_tokens == 7 and not live.cached
    store = CassetteStore(tmp_path / "c")
    ReplayClient(fake.client(), store, ReplayMode.RECORD).embed(request)
    replayed = ReplayClient(None, store, ReplayMode.REPLAY).embed(request)
    assert replayed.cached and replayed.embeddings == live.embeddings
    with pytest.raises(CassetteMissError):
        ReplayClient(None, store, ReplayMode.REPLAY).embed(
            request.model_copy(update={"input": ["x"]})
        )
    tracer = Tracer(tmp_path / "t.jsonl")
    TracedClient(fake.client(), tracer).embed(request)
    span = read_trace(tmp_path / "t.jsonl")[0]
    assert span["name"] == "llm.embed" and span["n_inputs"] == 1


def test_embedder_prefixes_batches_and_normalises(fake):
    embedder = Embedder(fake.client(), "embeddinggemma:latest", batch=2)
    vectors = embedder.documents(["blink detection", "eye blinks", "attention tokens"])
    assert vectors.shape == (3, 64) and np.allclose(np.linalg.norm(vectors, axis=1), 1)
    embeds = [b for p, b in fake.requests if p == "/api/embed"]
    assert len(embeds) == 2 and embeds[0]["input"][0].startswith("title: none | text: ")
    query = embedder.query("blink")
    assert fake.requests[-1][1]["input"] == ["task: search result | query: blink"]
    assert float(vectors[0] @ query) > float(vectors[2] @ query)
    assert Embedder(fake.client(), "other-model").query_prefix == ""
