"""Step 1 (ROUTING): classify the paper so the outline can use the right narrative."""

from __future__ import annotations

from paper2flow.llm.structured import structured_chat
from paper2flow.schemas import Paper, Route
from paper2flow.steps.llm import LLM, load_prompt

MIN_CONFIDENCE = 0.6
"""Below this the router falls back to the most general template (``method``)."""


def route_paper(paper: Paper, llm: LLM) -> Route:
    """Classify a paper as method, benchmark, survey or position.

    Only the title and abstract are sent: routing is cheap by design. Low-confidence
    decisions fall back to ``method``, and the fallback is recorded in ``reason``.

    Args:
        paper: Ingested paper.
        llm: Model settings.

    Returns:
        The routing decision.
    """
    prompt = load_prompt("route").format(title=paper.title, abstract=paper.abstract)
    route = structured_chat(llm.backend, llm.request(prompt), Route)
    if route.confidence < MIN_CONFIDENCE and route.paper_type != "method":
        return Route(
            paper_type="method",
            confidence=route.confidence,
            reason=f"fallback from {route.paper_type} (confidence {route.confidence:.2f}): "
            f"{route.reason}",
        )
    return route
