"""Typed configuration loaded from ``config.toml``."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config.toml")

PROFILE_ENV = "LABMATE_PROFILE"
"""Environment variable that picks a model profile (see ``[profiles.<name>]``)."""

LIBRARY_ENV = "LABMATE_LIBRARY"
"""Environment variable that overrides ``[ask].library`` (a private library outside the repo)."""

HOST_ENV = "LABMATE_OLLAMA_HOST"
"""Environment variable that overrides ``[ollama].host`` (e.g. Ollama on another machine)."""
"""Config file used when no explicit path is given (relative to the working directory)."""


class OllamaConfig(BaseModel):
    """Connection settings for the local Ollama server."""

    host: str = "http://localhost:11434"
    timeout_s: float = 900.0


class ModelsConfig(BaseModel):
    """Model tag for each pipeline role."""

    text: str = "gemma4:26b-mlx"
    critic: str = "gemma4:26b-mlx"
    decider: str = "clef-flash"
    embed: str = "embeddinggemma:latest"


class ProfileConfig(BaseModel):
    """A named model setup: which model plays the writer and which the judge.

    Attributes:
        text: Writer model (generation, agents).
        critic: Judge model (fact-check, grading, verification).
        embed: Embedding model; the ``[models]`` one if omitted.
        num_ctx: Context window; the ``[generation]`` one if omitted.
    """

    text: str
    critic: str
    embed: str | None = None
    num_ctx: int | None = None


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
    """Settings of the ask feature (questions about a library of documents).

    Attributes:
        library: Folder with ``library.toml`` and the source PDFs. ``LABMATE_LIBRARY`` and
            ``labmate ask --library`` replace it, so a private library can live outside the repo.
        index: The search index (SQLite: sections, chunks, full-text index, vectors); by default
            ``index.sqlite`` inside the library.
        chunk_words: Target words per chunk (paragraphs are packed up to this size).
        top_k: Chunks retrieved per search.
        max_loops: Retrieve-grade-rewrite rounds per question.
        embed_batch: Texts per embedding call.
        num_ctx: Context window for the ask models. The prompts are short, and a smaller window
            makes each model smaller in memory, so the writer and the judge stay loaded together.
    """

    library: Path = Path("library")
    index: Path | None = None
    chunk_words: int = 180
    top_k: int = 4
    max_loops: int = 2
    embed_batch: int = 32
    num_ctx: int = 8192

    @property
    def index_file(self) -> Path:
        """The index file: ``index`` if set, else ``index.sqlite`` inside the library."""
        return self.index or self.library / "index.sqlite"


class CacheConfig(BaseModel):
    """Reply cache settings.

    Attributes:
        enabled: Serve repeated requests from the cache.
        path: The SQLite file.
    """

    enabled: bool = True
    path: Path = Path("cache/replies.sqlite")


class TracingConfig(BaseModel):
    """Where run traces are written."""

    runs_dir: Path = Path("runs")


class PostConfig(BaseModel):
    """Settings of paper2post.

    Attributes:
        links: Your own links, shown under every post (label -> URL).
    """

    links: dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    """Top-level configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    ask: AskConfig = Field(default_factory=AskConfig)
    post: PostConfig = Field(default_factory=PostConfig)
    callbacks: list[Any] = Field(default_factory=list, exclude=True)
    """LangChain callbacks added to every chat model (the benchmark's usage collector)."""


def apply_profile(config: Config, name: str) -> None:
    """Switch the configuration to a named model profile, in place.

    Args:
        config: The configuration.
        name: A key of ``[profiles]``.

    Raises:
        ValueError: If there is no such profile.
    """
    if name not in config.profiles:
        known = ", ".join(sorted(config.profiles)) or "none defined"
        raise ValueError(f"unknown profile {name!r} (known: {known})")
    profile = config.profiles[name]
    config.models.text, config.models.critic = profile.text, profile.critic
    if profile.embed:
        config.models.embed = profile.embed
    if profile.num_ctx:
        config.generation.num_ctx = profile.num_ctx


def load_config(path: Path | None = None, profile: str | None = None) -> Config:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML file. Defaults to :data:`DEFAULT_CONFIG_PATH`. If the default
            file does not exist, built-in defaults are returned.
        profile: A model profile to apply (default: ``LABMATE_PROFILE``, else none).

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If an explicit ``path`` is given but does not exist.
        ValueError: If the profile is not defined.

    ``LABMATE_OLLAMA_HOST`` overrides the Ollama host and ``LABMATE_LIBRARY`` the ask library.
    """
    if path is None and not DEFAULT_CONFIG_PATH.exists():
        config = Config()
    else:
        with (path or DEFAULT_CONFIG_PATH).open("rb") as f:
            config = Config.model_validate(tomllib.load(f))
    if host := os.environ.get(HOST_ENV):
        config.ollama.host = host
    if library := os.environ.get(LIBRARY_ENV):
        config.ask.library = Path(library)
    if name := profile or os.environ.get(PROFILE_ENV):
        apply_profile(config, name)
    return config
