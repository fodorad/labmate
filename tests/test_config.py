from pathlib import Path

import pytest

from labmate.config import Config, load_config

REPO_CONFIG = Path(__file__).parent.parent / "config.toml"


def test_repo_config_is_valid_and_pins_the_roles():
    config = load_config(REPO_CONFIG)
    assert config.models.text == "qwen3.6:35b-mlx"
    assert config.models.critic == "gemma4:26b-mlx"
    assert config.cache.enabled
    assert config.generation.temperature == 0.0


def test_defaults_when_no_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config() == Config()


def test_default_path_is_used_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("[cache]\nenabled = false\n")
    assert load_config().cache.enabled is False


def test_explicit_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_an_invalid_value_is_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[cache]\nenabled = "sometimes"\n')
    with pytest.raises(ValueError):
        load_config(path)


def test_environment_overrides_the_ollama_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LABMATE_OLLAMA_HOST", "http://192.168.0.2:11434")
    assert load_config().ollama.host == "http://192.168.0.2:11434"
    assert load_config(REPO_CONFIG).ollama.host == "http://192.168.0.2:11434"
