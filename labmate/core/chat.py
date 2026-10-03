"""Chat models and embeddings: LangChain's Ollama integration plus a persistent reply cache.

Every feature gets its models from here, so sampling settings, the server address and the
cache are set in one place. A repeated request (same messages, model and options) is served
from the SQLite cache instead of the model, which makes reruns instant and keeps a pipeline
deterministic once it has run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from functools import cache as memoize
from pathlib import Path
from typing import Any

import httpx
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration
from langchain_ollama import ChatOllama, OllamaEmbeddings

from labmate.config import Config


class SQLiteCache(BaseCache):
    """A LangChain cache of chat replies in one SQLite file.

    Only chat generations are stored (that is all labmate caches). One connection is shared
    by the worker threads of parallel steps, guarded by a lock.

    Args:
        path: The SQLite file (created with its folder if missing).
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                "create table if not exists replies"
                " (prompt text, llm text, reply text, primary key (prompt, llm))"
            )

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """Return the cached generations of a request, or ``None`` on a miss.

        Args:
            prompt: The serialised messages.
            llm_string: The model and its options.

        Returns:
            The generations, or ``None``.
        """
        with self._lock:
            row = self._db.execute(
                "select reply from replies where prompt = ? and llm = ?", (prompt, llm_string)
            ).fetchone()
        if row is None:
            return None
        return [
            ChatGeneration(message=message) for message in messages_from_dict(json.loads(row[0]))
        ]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Store the generations of a request.

        Args:
            prompt: The serialised messages.
            llm_string: The model and its options.
            return_val: The generations to store.
        """
        messages = [message_to_dict(g.message) for g in return_val if isinstance(g, ChatGeneration)]
        with self._lock, self._db:
            self._db.execute(
                "insert or replace into replies values (?, ?, ?)",
                (prompt, llm_string, json.dumps(messages)),
            )

    def close(self) -> None:
        """Close the SQLite file."""
        with self._lock:
            self._db.close()

    def clear(self, **kwargs: Any) -> None:
        """Delete every cached reply."""
        with self._lock, self._db:
            self._db.execute("delete from replies")


class OllamaChat(ChatOllama):
    """:class:`ChatOllama` whose cache key holds the whole request.

    The stock key is only the class name and stop words, so a cache shared between models,
    seeds or output schemas would replay a reply made for another request. This key adds the
    model and its options (seed, temperature, context window), the ``format`` constraint
    and the tools.
    """

    def _get_invocation_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        request = self._chat_params([], stop, **kwargs)
        request.pop("messages")
        return {**super()._get_invocation_params(stop, **kwargs), **request}


def chat_model(
    config: Config,
    model: str,
    *,
    cache: BaseCache | None = None,
    transport: httpx.BaseTransport | None = None,
    json_schema: dict[str, Any] | None = None,
) -> OllamaChat:
    """A chat model on the local Ollama server with the configured sampling settings.

    Thinking is off and sampling is deterministic (temperature and seed from the config).

    Args:
        config: Loaded configuration.
        model: Model tag, e.g. ``config.models.text``.
        cache: Reply cache; the configured SQLite file when omitted (none if the config
            switches caching off).
        transport: An httpx transport to reach the server through (tests).
        json_schema: JSON schema sent as Ollama's ``format`` constraint.

    Returns:
        The chat model.
    """
    generation = config.generation
    return OllamaChat(
        model=model,
        base_url=config.ollama.host,
        temperature=generation.temperature,
        seed=generation.seed,
        num_ctx=generation.num_ctx,
        reasoning=False,
        format=json_schema,
        client_kwargs={"timeout": config.ollama.timeout_s, "transport": transport},
        cache=_cache(config) if cache is None else cache,
        disable_streaming=True,
    )


def embedder(config: Config, *, transport: httpx.BaseTransport | None = None) -> OllamaEmbeddings:
    """The embedding model on the local Ollama server.

    Args:
        config: Loaded configuration.
        transport: An httpx transport to reach the server through (tests).

    Returns:
        The embeddings.
    """
    return OllamaEmbeddings(
        model=config.models.embed,
        base_url=config.ollama.host,
        client_kwargs={"timeout": config.ollama.timeout_s, "transport": transport},
    )


@memoize
def _open_cache(path: Path) -> SQLiteCache:
    """One cache connection per file (keyed by its absolute path), shared by every model."""
    return SQLiteCache(path)


def _cache(config: Config) -> BaseCache | bool:
    """The configured cache, or ``False`` when caching is off."""
    return _open_cache(config.cache.path.resolve()) if config.cache.enabled else False
