"""M1 walking skeleton: one structured LLM call from paper to deck.

This is deliberately the naive baseline. Later milestones replace it with routing,
parallel claim extraction, an orchestrated outline and a fact-check loop, and the
evaluation compares against this one-shot version.
"""

from __future__ import annotations

from importlib.resources import files

from paper2carousel.llm.client import Backend
from paper2carousel.llm.structured import structured_chat
from paper2carousel.llm.types import ChatRequest, Message
from paper2carousel.schemas import Deck, Paper

PROMPT = files("paper2carousel.prompts").joinpath("draft.md").read_text()
"""Prompt template for the one-shot draft (versioned with the code)."""

SECTION_CHARS = 1500
"""Characters kept from the start of each section; keeps the prompt within ~6k tokens."""


def build_prompt(paper: Paper, n_min: int = 6, n_max: int = 8) -> str:
    """Fill the draft prompt with the paper's abstract and truncated sections.

    Args:
        paper: Ingested paper.
        n_min: Minimum slide count.
        n_max: Maximum slide count.

    Returns:
        The user prompt.
    """
    sections = "\n\n".join(
        f"## {s.title} (page {s.page})\n{s.text[:SECTION_CHARS]}" for s in paper.sections
    )
    return PROMPT.format(
        n_min=n_min, n_max=n_max, title=paper.title, abstract=paper.abstract, sections=sections
    )


def draft_deck(
    paper: Paper,
    backend: Backend,
    model: str,
    seed: int = 42,
    temperature: float = 0.0,
    num_ctx: int = 16384,
) -> Deck:
    """Draft a deck in a single structured call.

    Args:
        paper: Ingested paper.
        backend: Model backend.
        model: Text model tag.
        seed: Sampling seed.
        temperature: Sampling temperature.
        num_ctx: Context window; the default Ollama window would silently truncate the paper.

    Returns:
        The drafted deck.
    """
    request = ChatRequest(
        model=model,
        messages=[Message(role="user", content=build_prompt(paper))],
        seed=seed,
        temperature=temperature,
        num_ctx=num_ctx,
        think=False,
    )
    return structured_chat(backend, request, Deck)
