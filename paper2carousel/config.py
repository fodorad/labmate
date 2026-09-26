"""Typed configuration loaded from ``config.toml``."""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config.toml")
"""Config file used when no explicit path is given (relative to the working directory)."""


class ReplayMode(StrEnum):
    """How model calls interact with the cassette store.

    Attributes:
        LIVE: Always call Ollama and store nothing.
        RECORD: Always call Ollama and (over)write cassettes.
        AUTO: Use a cassette if present, otherwise call Ollama and record it.
        REPLAY: Cassettes only; a missing cassette is an error.
    """

    LIVE = "live"
    RECORD = "record"
    AUTO = "auto"
    REPLAY = "replay"


class OllamaConfig(BaseModel):
    """Connection settings for the local Ollama server."""

    host: str = "http://localhost:11434"
    timeout_s: float = 900.0


class ModelsConfig(BaseModel):
    """Model tag for each pipeline role."""

    text: str = "qwen3.6:35b-mlx"
    critic: str = "gemma4:26b-mlx"
    vision: str = "gemma4:e4b"
    image: str = "x/z-image-turbo:latest"
    image_candidates: list[str] = Field(
        default_factory=lambda: ["x/z-image-turbo:latest", "x/flux2-klein:latest"]
    )


class GenerationConfig(BaseModel):
    """Default sampling options that make runs deterministic."""

    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 16384


class PipelineConfig(BaseModel):
    """Orchestration settings."""

    workers: int = 2
    max_rewrite_rounds: int = 2


class VisualsConfig(BaseModel):
    """Visuals settings."""

    enabled: bool = True
    cover_image: bool = True
    critic: bool = True


class OutputsConfig(BaseModel):
    """Extra outputs next to the carousel."""

    post: bool = True


class ReplayConfig(BaseModel):
    """Record/replay cache settings."""

    mode: ReplayMode = ReplayMode.AUTO
    dir: Path = Path("cassettes")
    lock_file: Path = Path("models.lock")


class TracingConfig(BaseModel):
    """Where run traces are written."""

    runs_dir: Path = Path("runs")


class Config(BaseModel):
    """Top-level configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    visuals: VisualsConfig = Field(default_factory=VisualsConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)


def load_config(path: Path | None = None) -> Config:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML file. Defaults to :data:`DEFAULT_CONFIG_PATH`. If the default
            file does not exist, built-in defaults are returned.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If an explicit ``path`` is given but does not exist.
    """
    if path is None:
        if not DEFAULT_CONFIG_PATH.exists():
            return Config()
        path = DEFAULT_CONFIG_PATH
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))
