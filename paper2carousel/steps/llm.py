"""Shared model-call settings for the pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from paper2carousel.llm.client import Backend
from paper2carousel.llm.types import ChatRequest, Message


def load_prompt(name: str) -> str:
    """Read a versioned prompt template shipped in ``paper2carousel/prompts``.

    Args:
        name: File stem, e.g. ``"route"``.

    Returns:
        The template text (``str.format`` placeholders).
    """
    return files("paper2carousel.prompts").joinpath(f"{name}.md").read_text()


@dataclass(frozen=True)
class LLM:
    """A backend plus the sampling settings every step uses.

    Attributes:
        backend: Traced, replayed backend.
        model: Model tag for this role.
        seed: Sampling seed.
        temperature: Sampling temperature.
        num_ctx: Context window.
    """

    backend: Backend
    model: str
    seed: int = 42
    temperature: float = 0.0
    num_ctx: int = 16384

    def request(self, prompt: str) -> ChatRequest:
        """Build a single-turn request with thinking disabled.

        Args:
            prompt: User message.

        Returns:
            The request.
        """
        return ChatRequest(
            model=self.model,
            messages=[Message(role="user", content=prompt)],
            seed=self.seed,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            think=False,
        )
