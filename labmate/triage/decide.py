"""The decision of one paper: read it deeply, make a post of it, or skip it.

Two kinds of model can be the ``decider``, behind one :class:`Decider` interface. A decision model
(``clef-flash``, ``nimble``, ``tev1``) scores the paper on a five-level relevance scale in one pass
and code turns the score into an action. An ordinary chat model is asked for the action, the
relevance and a reason as validated JSON. :func:`open_decider` asks Ollama once which kind a model
is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field

from labmate.config import Config
from labmate.core.chat import chat_model
from labmate.core.ingest import ArxivMetadata
from labmate.core.lc import prompt, structured
from labmate.core.ollama import OllamaServer
from labmate.triage.prompts import load_prompt

DECISION = "decision"
"""The capability Ollama lists for a decision model."""

MAX_REASON_WORDS = 25
"""Longest reason a chat model may give, in words."""

LEVELS = [
    "unrelated",
    "slightly related",
    "related to a neighbouring area",
    "closely related",
    "exactly on the interests",
]
"""The relevance scale a decision model scores on, from low to high (levels 0 to 4)."""

DEEP_AT = 3.0
"""Score from which a paper gets a deep read (level 3, closely related, or above)."""

POST_AT = 1.75
"""Score from which a paper is worth a post (about a neighbouring area, or above)."""

QUESTION = {
    "type": "score",
    "instructions": "How relevant is this paper to the reader's interests?",
    "criteria": LEVELS,
}
"""The one question a decision model answers about a paper."""


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


def action_for(score: float) -> Literal["deep", "post", "skip"]:
    """The action a relevance score calls for.

    Args:
        score: A decision model's score, from 0 to 4.

    Returns:
        ``deep`` from :data:`DEEP_AT`, ``post`` from :data:`POST_AT`, else ``skip``.
    """
    if score >= DEEP_AT:
        return "deep"
    return "post" if score >= POST_AT else "skip"


def decision_from_score(score: float) -> Decision:
    """A decision from a decision model's relevance score.

    Args:
        score: The score, from 0 to 4.

    Returns:
        The decision; its reason names the nearest level and the score.
    """
    level = min(max(round(score), 0), len(LEVELS) - 1)
    return Decision(
        action=action_for(score),
        relevance=level + 1,
        reason=f"{LEVELS[level]} (score {score:.1f} of {len(LEVELS) - 1})",
    )


class Decider(Protocol):
    """Rates papers against the reader's interests."""

    @property
    def kind(self) -> str:
        """``decision`` (a decision model) or ``chat`` (a chat model)."""
        ...

    def decide(
        self, meta: ArxivMetadata, interests: str, run_config: RunnableConfig | None = None
    ) -> Decision:
        """Rate one paper.

        Args:
            meta: The paper's title and abstract.
            interests: What the reader works on.
            run_config: The calling step's config (carries the callbacks).

        Returns:
            The validated decision.
        """
        ...


@dataclass(frozen=True)
class DecisionModelDecider:
    """A decision model scores the paper in one pass; code turns the score into an action.

    Attributes:
        server: The Ollama server it runs on.
        model: Model tag.
        kind: Always ``decision``.
    """

    server: OllamaServer
    model: str
    kind: str = DECISION

    def decide(
        self, meta: ArxivMetadata, interests: str, run_config: RunnableConfig | None = None
    ) -> Decision:
        """Rate one paper (see :class:`Decider`)."""
        state = (
            f"Reader's interests: {interests}\n\n"
            f"Paper title: {meta.title}\nAbstract: {meta.abstract}"
        )
        answers = self.server.systemone(self.model, state, {"relevance": QUESTION})
        return decision_from_score(float(answers["relevance"]["score"]))


@dataclass(frozen=True)
class ChatDecider:
    """A chat model is asked for the action, the relevance and a reason as validated JSON.

    Attributes:
        chain: The prompt piped into the structured model.
        kind: Always ``chat``.
    """

    chain: Runnable[dict[str, Any], Decision]
    kind: str = "chat"

    def decide(
        self, meta: ArxivMetadata, interests: str, run_config: RunnableConfig | None = None
    ) -> Decision:
        """Rate one paper (see :class:`Decider`)."""
        values = {
            "interests": interests,
            "title": meta.title,
            "abstract": meta.abstract,
            "max_words": MAX_REASON_WORDS,
        }
        return self.chain.invoke(values, run_config)


def open_decider(
    config: Config, model: str | None = None, transport: httpx.BaseTransport | None = None
) -> Decider:
    """The decider for a model; which kind it is comes from the capabilities Ollama lists.

    Args:
        config: Loaded configuration.
        model: Model tag (default: the ``decider`` model role).
        transport: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        A :class:`DecisionModelDecider` or a :class:`ChatDecider`.
    """
    tag = model or config.models.decider
    server = OllamaServer.from_config(config, transport)
    if DECISION in server.capabilities(tag):
        return DecisionModelDecider(server, tag)
    chat = chat_model(config, tag, transport=transport)
    return ChatDecider(
        prompt(load_prompt("triage")) | structured(chat, Decision, check=reason_problems)
    )
