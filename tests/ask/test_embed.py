import numpy as np

from labmate.ask.embed import Embedder
from labmate.core.chat import embedder


def test_documents_are_prefixed_batched_and_normalised(fake, config):
    model = Embedder(embedder(config, transport=fake.transport()), "embeddinggemma:latest", batch=2)

    vectors = model.documents(["blink detection", "eye blinks", "attention tokens"])

    assert vectors.shape == (3, 64) and np.allclose(np.linalg.norm(vectors, axis=1), 1)
    sent = [b for p, b in fake.requests if p == "/api/embed"]
    assert len(sent) == 2 and sent[0]["input"][0].startswith("title: none | text: ")


def test_a_query_is_closest_to_the_documents_that_share_its_words(fake, config):
    model = Embedder(embedder(config, transport=fake.transport()), "embeddinggemma:latest")
    vectors = model.documents(["blink detection", "attention tokens"])

    query = model.query("blink")

    assert fake.requests[-1][1]["input"] == ["task: search result | query: blink"]
    assert float(vectors[0] @ query) > float(vectors[1] @ query)


def test_models_without_known_prefixes_get_none(fake, config):
    assert Embedder(embedder(config, transport=fake.transport()), "other-model").query_prefix == ""
