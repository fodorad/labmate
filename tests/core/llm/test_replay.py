import pytest

from labmate.config import ReplayMode
from labmate.core.llm.replay import (
    CassetteMissError,
    CassetteStore,
    ReplayClient,
    read_lock,
    write_lock,
)
from labmate.core.llm.types import ChatRequest, Message

REQ = ChatRequest(model="qwen3.6:35b-mlx", messages=[Message(role="user", content="hi")])


def n_calls(fake):
    return fake.paths().count("/api/chat")


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


def test_lock_roundtrip(tmp_path):
    path = tmp_path / "models.lock"
    assert read_lock(path) == {}
    write_lock(path, {"b:1": "d2", "a:1": "d1"})
    assert read_lock(path) == {"a:1": "d1", "b:1": "d2"}
    assert path.read_text().index("a:1") < path.read_text().index("b:1")
