"""Everything an ask command needs: the recorded backend, the model roles, the index."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from labmate.ask.embed import Embedder
from labmate.ask.index import Index
from labmate.config import Config, ReplayMode
from labmate.core.llm.client import OllamaClient
from labmate.core.llm.replay import CassetteStore, ReplayClient, read_lock
from labmate.core.model import LLM
from labmate.core.phases import ModelSwitcher
from labmate.core.tracing import TracedClient, Tracer


@dataclass
class AskSession:
    """Shared state of an ask command.

    Attributes:
        config: Loaded configuration.
        tracer: Span recorder (``<library>/trace.jsonl``).
        llm: Writer / planner settings.
        judge: Grader and fact-check judge settings.
        embedder: Query and document embeddings.
        switcher: Keeps one large model resident at a time.
        index: The search index.
        mode: Replay mode in effect.
        connections: Extra databases to close with the session (conversation checkpoints).
    """

    config: Config
    tracer: Tracer
    llm: LLM
    judge: LLM
    embedder: Embedder
    switcher: ModelSwitcher
    index: Index
    mode: ReplayMode
    connections: list[sqlite3.Connection] = field(default_factory=list)

    @property
    def workers(self) -> int:
        """Concurrent model calls within a phase."""
        return self.config.pipeline.workers

    def close(self) -> None:
        """Release the models and close the index."""
        self.switcher.release()
        self.index.close()
        for connection in self.connections:
            connection.close()


def open_ask(
    config: Config,
    mode: ReplayMode | None = None,
    client: OllamaClient | None = None,
    trace: Path | None = None,
    label: str = "ask",
) -> AskSession:
    """Open the index and build the recorded, traced backend.

    Args:
        config: Loaded configuration.
        mode: Replay mode override.
        client: Ollama client (built from config if omitted; unused in replay mode).
        trace: Trace file (default: ``<library>/trace.jsonl``).
        label: Prefix of the trace id.

    Returns:
        The session.
    """
    mode = mode or config.replay.mode
    live = None
    if mode is not ReplayMode.REPLAY:
        live = client or OllamaClient(config.ollama.host, config.ollama.timeout_s)
    started = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    tracer = Tracer(
        trace or config.ask.library / "trace.jsonl",
        trace_id=f"{label}@{started}-{uuid.uuid4().hex[:6]}",
    )
    digests = read_lock(config.replay.lock_file)
    backend = TracedClient(
        ReplayClient(live, CassetteStore(config.replay.dir), mode, digests), tracer, digests
    )
    gen, m = config.generation, config.models
    return AskSession(
        config=config,
        tracer=tracer,
        llm=LLM(backend, m.text, gen.seed, gen.temperature, gen.num_ctx),
        judge=LLM(backend, m.critic, gen.seed, gen.temperature, gen.num_ctx),
        embedder=Embedder(backend, m.embed, config.ask.embed_batch),
        switcher=ModelSwitcher(live),
        index=Index(config.ask.index),
        mode=mode,
    )
