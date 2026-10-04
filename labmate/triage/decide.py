"""The decision of one paper: read it deeply, make a post of it, or skip it.

Two kinds of model can be the ``decider``. A decision model (``clef-flash``, ``nimble``,
``tev1``) scores the paper on a five-level relevance scale in one pass and code turns the score
into an action. An ordinary chat model is asked for the action, the relevance and a reason as
validated JSON. Which kind it is comes from the capabilities Ollama lists for it.
"""

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
from labmate.triage.systemone import has_decision_capability, systemone

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


def is_decision_model(config: Config, transport: httpx.BaseTransport | None = None) -> bool:
    """Whether the configured decider is a decision model.

    Args:
        config: Loaded configuration (the ``decider`` model role).
        transport: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        ``True`` if Ollama lists the ``decision`` capability for it.
    """
    return has_decision_capability(config, config.models.decider, transport)


def decide(
    config: Config,
    meta: ArxivMetadata,
    interests: str,
    *,
    decision_model: bool,
    run_config: RunnableConfig | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Decision:
    """Let the decider model rate one paper against the reader's interests.

    Args:
        config: Loaded configuration (the ``decider`` model role).
        meta: The paper's title and abstract.
        interests: What the reader works on.
        decision_model: Whether the decider is a decision model (see :func:`is_decision_model`).
        run_config: The calling step's config (carries the callbacks).
        transport: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        The validated decision.
    """
    if decision_model:
        state = (
            f"Reader's interests: {interests}\n\n"
            f"Paper title: {meta.title}\nAbstract: {meta.abstract}"
        )
        answers = systemone(
            config, config.models.decider, state, {"relevance": QUESTION}, transport
        )
        return decision_from_score(float(answers["relevance"]["score"]))
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
