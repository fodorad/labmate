"""Everything an ask command needs: the models, the embedder and the index."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from langchain_core.language_models import BaseChatModel

from labmate.ask.embed import Embedder
from labmate.ask.index import Index
from labmate.config import Config
from labmate.core.chat import chat_model, embedder


@dataclass
class AskSession:
    """Shared state of an ask command.

    Attributes:
        config: Loaded configuration.
        writer: Model that reads questions, rewrites queries and writes answers.
        judge: Model that grades the retrieved chunks.
        embedder: Query and document embeddings.
        index: The search index.
    """

    config: Config
    writer: BaseChatModel
    judge: BaseChatModel
    embedder: Embedder
    index: Index

    def close(self) -> None:
        """Close the index."""
        self.index.close()


def open_ask(config: Config, ollama: httpx.BaseTransport | None = None) -> AskSession:
    """Open the index and build the models.

    Args:
        config: Loaded configuration.
        ollama: Transport to the Ollama server (the configured host if omitted; tests).

    Returns:
        The session.
    """
    models = config.models
    return AskSession(
        config=config,
        writer=chat_model(config, models.text, transport=ollama),
        judge=chat_model(config, models.critic, transport=ollama),
        embedder=Embedder(embedder(config, transport=ollama), models.embed, config.ask.embed_batch),
        index=Index(config.ask.index),
    )
