"""ROUTING: classify the paper, so the cards and the flow are read the right way."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from labmate.core.lc import prompt, structured
from labmate.paper2flow.prompts import load_prompt
from labmate.paper2flow.schemas import Paper, Route

MIN_CONFIDENCE = 0.6
"""Below this the router falls back to the most general paper type (``method``)."""


def route_paper(paper: Paper, model: BaseChatModel, config: RunnableConfig | None = None) -> Route:
    """Classify a paper as method, benchmark, survey or position.

    Only the title and abstract are sent: routing is cheap by design. Low-confidence
    decisions fall back to ``method``, and the fallback is recorded in ``reason``.

    Args:
        paper: Ingested paper.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        The routing decision.
    """
    chain = prompt(load_prompt("route")) | structured(model, Route)
    route = chain.invoke({"title": paper.title, "abstract": paper.abstract}, config)
    if route.confidence < MIN_CONFIDENCE and route.paper_type != "method":
        return Route(
            paper_type="method",
            confidence=route.confidence,
            reason=f"fallback from {route.paper_type} (confidence {route.confidence:.2f}): "
            f"{route.reason}",
        )
    return route
