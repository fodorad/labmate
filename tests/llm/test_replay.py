import json

import pytest

from paper2flow.config import ReplayMode
from paper2flow.llm.replay import (
    CassetteMissError,
    CassetteStore,
    ReplayClient,
    read_lock,
    write_lock,
)
from paper2flow.llm.types import ChatRequest, ImageRequest, Message

REQ = ChatRequest(model="qwen3.6:35b-mlx", messages=[Message(role="user", content="hi")])
IMG = ImageRequest(model="x/flux2-klein:latest", prompt="p", seed=3)


def n_calls(fake, path="/api/chat"):
    return fake.paths().count(path)


def test_store_roundtrip_and_sharding(tmp_path):
    store = CassetteStore(tmp_path)
    assert store.get("ab" + "0" * 62) is None
    store.put("ab" + "0" * 62, {"q": 1}, {"a": 2})
    assert store.get("ab" + "0" * 62) == {"a": 2}
    record = json.loads(store.path("ab" + "0" * 62).read_text())
    assert store.path("ab" + "0" * 62).parent.name == "ab"
    assert record["request"] == {"q": 1}


def test_live_mode_never_stores(fake, tmp_path):
    client = ReplayClient(fake.client(), CassetteStore(tmp_path), ReplayMode.LIVE)
    client.chat(REQ)
    client.chat(REQ)
    assert n_calls(fake) == 2
    assert not any(tmp_path.rglob("*.json"))


def test_auto_mode_records_then_replays(fake, tmp_path):
    client = ReplayClient(fake.client(), CassetteStore(tmp_path), ReplayMode.AUTO)
    first = client.chat(REQ)
    second = client.chat(REQ)
    assert n_calls(fake) == 1
    assert first.cached is False and second.cached is True
    assert first.content == second.content
    assert first.usage == second.usage


def test_record_mode_always_calls_and_overwrites(fake, tmp_path):
    store = CassetteStore(tmp_path)
    client = ReplayClient(fake.client(), store, ReplayMode.RECORD)
    client.chat(REQ)
    fake.chat_handler = lambda body: {"model": body["model"], "message": {"content": "new"}}
    assert client.chat(REQ).content == "new"
    assert n_calls(fake) == 2
    assert store.get(REQ.cache_key())["content"] == "new"


def test_replay_mode_works_without_backend_and_fails_on_miss(fake, tmp_path):
    store = CassetteStore(tmp_path)
    ReplayClient(fake.client(), store, ReplayMode.RECORD).chat(REQ)
    offline = ReplayClient(None, store, ReplayMode.REPLAY)
    assert offline.chat(REQ).cached is True
    other = ChatRequest(model=REQ.model, messages=[Message(role="user", content="other")])
    with pytest.raises(CassetteMissError):
        offline.chat(other)


def test_non_replay_mode_requires_backend(tmp_path):
    with pytest.raises(ValueError, match="needs a live backend"):
        ReplayClient(None, CassetteStore(tmp_path), ReplayMode.AUTO)


def test_digest_pins_are_part_of_the_key(fake, tmp_path):
    store = CassetteStore(tmp_path)
    ReplayClient(fake.client(), store, ReplayMode.RECORD, {REQ.model: "sha256:v1"}).chat(REQ)
    new_version = ReplayClient(None, store, ReplayMode.REPLAY, {REQ.model: "sha256:v2"})
    with pytest.raises(CassetteMissError):
        new_version.chat(REQ)


def test_image_record_and_replay(fake, tmp_path):
    store = CassetteStore(tmp_path)
    recorded = ReplayClient(fake.client(), store, ReplayMode.AUTO).generate_image(IMG)
    replayed = ReplayClient(None, store, ReplayMode.REPLAY).generate_image(IMG)
    assert replayed.cached is True
    assert replayed.sha256() == recorded.sha256()
    assert n_calls(fake, "/api/generate") == 1


def test_lock_roundtrip(tmp_path):
    path = tmp_path / "models.lock"
    assert read_lock(path) == {}
    write_lock(path, {"b:1": "d2", "a:1": "d1"})
    assert read_lock(path) == {"a:1": "d1", "b:1": "d2"}
    assert path.read_text().index("a:1") < path.read_text().index("b:1")
