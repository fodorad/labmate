"""The decision of one paper: read it deeply, make a post of it, or skip it."""

from __future__ import annotations

from typing import Literal

import httpx
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from labmate.config import Config
from labmate.core.chat import chat_model
from labmate.core.ingest import ArxivMetadata
from labmate.core.lc import prompt, structured
from labmate.triage.prompts import load_prompt

MAX_REASON_WORDS = 25
"""Longest reason the model may give, in words."""


class Decision(BaseModel):
    """What to do with one paper.

    Attributes:
        action: ``deep`` (full overview), ``post`` (a short post) or ``skip``.
        relevance: How much the paper matters to the interests, 1 (unrelated) to 5.
        reason: One short sentence.
    """

    action: Literal["deep", "post", "skip"]
    relevance: int = Field(ge=1, le=5)
    reason: str


def reason_problems(decision: Decision) -> list[str]:
    """Rules the schema cannot express: a reason is one short sentence.

    Args:
        decision: A schema-valid decision.

    Returns:
        The problems found (empty if none), sent back to the model.
    """
    words = len(decision.reason.split())
    if words > MAX_REASON_WORDS:
        return [f"the reason has {words} words; use at most {MAX_REASON_WORDS}"]
    return []


def decide(
    config: Config,
    meta: ArxivMetadata,
    interests: str,
    *,
    run_config: RunnableConfig | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Decision:
    """Let the decider model rate one paper against the reader's interests.

    Args:
        config: Loaded configuration (the ``decider`` model role).
        meta: The paper's title and abstract.
        interests: What the reader works on.
        run_config: The calling step's config (carries the callbacks).
        transport: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        The validated decision.
    """
    model = chat_model(config, config.models.decider, transport=transport)
    chain = prompt(load_prompt("triage")) | structured(model, Decision, check=reason_problems)
    return chain.invoke(
        {
            "interests": interests,
            "title": meta.title,
            "abstract": meta.abstract,
            "max_words": MAX_REASON_WORDS,
        },
        run_config,
    )
