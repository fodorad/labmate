"""Typed configuration loaded from ``config.toml``."""

from __future__ import annotations

import os
import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config.toml")

HOST_ENV = "LABMATE_OLLAMA_HOST"
"""Environment variable that overrides ``[ollama].host`` (e.g. Ollama on another machine)."""
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
    embed: str = "embeddinggemma:latest"


class GenerationConfig(BaseModel):
    """Default sampling options that make runs deterministic."""

    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 16384


class PipelineConfig(BaseModel):
    """Orchestration settings."""

    workers: int = 2
    max_rewrite_rounds: int = 2


class AskConfig(BaseModel):
    """Settings of the ask feature (questions about your research).

    Attributes:
        library: Folder with ``library.toml`` and the source PDFs.
        index: The search index (SQLite: sections, chunks, full-text index, vectors).
        threads: Conversation checkpoints (LangGraph ``SqliteSaver``).
        chunk_words: Target words per chunk (paragraphs are packed up to this size).
        top_k: Chunks retrieved per search.
        max_loops: Retrieve-grade-rewrite rounds per sub-question.
        embed_batch: Texts per embedding call.
        summary_min_words: Chapters shorter than this are not summarised (their chunks
            already say it all).
    """

    library: Path = Path("library")
    index: Path = Path("library/index.sqlite")
    threads: Path = Path("library/threads.sqlite")
    chunk_words: int = 180
    top_k: int = 6
    max_loops: int = 2
    embed_batch: int = 32
    summary_min_words: int = 300


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
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    ask: AskConfig = Field(default_factory=AskConfig)


def load_config(path: Path | None = None) -> Config:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML file. Defaults to :data:`DEFAULT_CONFIG_PATH`. If the default
            file does not exist, built-in defaults are returned.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If an explicit ``path`` is given but does not exist.

    ``LABMATE_OLLAMA_HOST`` overrides the Ollama host.
    """
    if path is None and not DEFAULT_CONFIG_PATH.exists():
        config = Config()
    else:
        with (path or DEFAULT_CONFIG_PATH).open("rb") as f:
            config = Config.model_validate(tomllib.load(f))
    if host := os.environ.get(HOST_ENV):
        config.ollama.host = host
    return config
