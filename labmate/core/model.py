"""Model-call settings shared by every step, and prompt loading."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files

from labmate.core.llm.client import Backend
from labmate.core.llm.types import ChatRequest, Message


def load_prompt(name: str, package: str) -> str:
    """Read a versioned prompt template shipped as ``<package>/<name>.md``.

    Args:
        name: File stem, e.g. ``"route"``.
        package: Package holding the prompts, e.g. ``"labmate.paper2flow.prompts"``.

    Returns:
        The template text (``str.format`` placeholders).
    """
    return files(package).joinpath(f"{name}.md").read_text()


def prompt_loader(package: str) -> Callable[[str], str]:
    """A :func:`load_prompt` bound to one feature's prompt package.

    Args:
        package: Package holding the prompts.

    Returns:
        ``load(name) -> template``.
    """

    def load(name: str) -> str:
        return load_prompt(name, package)

    return load


@dataclass(frozen=True)
class LLM:
    """A backend plus the sampling settings every step uses.

    Attributes:
        backend: Traced, replayed backend.
        model: Model tag for this role.
        seed: Sampling seed.
        temperature: Sampling temperature.
        num_ctx: Context window.
        digest: Pinned digest of ``model`` (from ``models.lock``); part of every cache key,
            so a new model version never replays an old answer.
    """

    backend: Backend
    model: str
    seed: int = 42
    temperature: float = 0.0
    num_ctx: int = 16384
    digest: str | None = None

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
