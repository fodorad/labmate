from pathlib import Path

import pytest

from labmate.config import Config, load_config

REPO_CONFIG = Path(__file__).parent.parent / "config.toml"


def test_repo_config_is_valid_and_pins_the_roles():
    config = load_config(REPO_CONFIG)
    assert config.models.text == "gemma4:26b-mlx"
    assert config.models.critic == "gemma4:e4b"
    assert config.cache.enabled
    assert config.generation.temperature == 0.0


def test_defaults_when_no_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config() == Config()


def test_environment_overrides_the_ollama_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LABMATE_OLLAMA_HOST", "http://192.168.0.2:11434")
    assert load_config().ollama.host == "http://192.168.0.2:11434"
    assert load_config(REPO_CONFIG).ollama.host == "http://192.168.0.2:11434"


def test_a_profile_swaps_the_writer_and_the_judge(tmp_path, monkeypatch):
    path = tmp_path / "c.toml"
    path.write_text('[profiles.small]\ntext = "w"\ncritic = "j"\nnum_ctx = 8192\n')

    config = load_config(path, "small")

    assert (config.models.text, config.models.critic) == ("w", "j")
    assert config.generation.num_ctx == 8192 and config.models.embed == Config().models.embed
    monkeypatch.setenv("LABMATE_PROFILE", "small")
    assert load_config(path).models.text == "w"  # the environment picks it too
    with pytest.raises(ValueError, match="unknown profile 'big' \\(known: small\\)"):
        load_config(path, "big")
