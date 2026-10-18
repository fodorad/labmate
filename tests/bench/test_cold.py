import json

import pytest

from labmate.bench.cold import ColdRun
from labmate.bench.guard import NotEnoughMemory
from labmate.config import Config
from labmate.core.chat import chat_model


@pytest.fixture
def cold(fake, tmp_path):
    return ColdRun(
        Config(), tmp_path / "results.json", transport=fake.transport(), ram_gb=32,
        free_gb=lambda: 30.0,
    )  # fmt: skip


def test_a_measurement_runs_cold_without_the_cache_and_is_saved_at_once(fake, cold):
    def measure(config, collector):
        chat_model(config, "gemma4:e4b", transport=fake.transport()).invoke("hi")
        return {"cache": config.cache.enabled, "calls": collector.summary().calls}

    cold.run("gemma4:e4b", 6.6, measure)

    saved = json.loads(cold.file.read_text())
    assert saved == {"gemma4:e4b": {"cache": False, "calls": 1}} == cold.results()
    assert fake.loaded == set()  # unloaded when it ended


def test_a_saved_result_is_not_measured_again_unless_forced(fake, cold):
    cold.file.write_text(json.dumps({"a": {"n": 1}}))
    calls = []

    cold.run("a", 1.0, lambda config, collector: calls.append(1) or {"n": 2})

    assert calls == [] and fake.requests == []
    cold.force = True
    cold.run("a", 1.0, lambda config, collector: {"n": 2})
    assert cold.results() == {"a": {"n": 2}}


def test_models_that_do_not_fit_are_refused_before_anything_loads(fake, cold):
    with pytest.raises(NotEnoughMemory):
        cold.run("big", 30.0, lambda config, collector: {})

    assert "/api/chat" not in fake.paths() and not cold.file.exists()


def test_models_are_unloaded_even_when_a_measurement_fails(fake, cold):
    def measure(config, collector):
        chat_model(config, "gemma4:e4b", transport=fake.transport()).invoke("hi")
        raise RuntimeError("the model broke")

    with pytest.raises(RuntimeError):
        cold.run("gemma4:e4b", 6.6, measure)

    assert fake.loaded == set() and cold.results() == {}
