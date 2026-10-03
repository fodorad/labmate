"""Prompt templates: Markdown files next to the code, loaded by package."""

from __future__ import annotations

from collections.abc import Callable
from importlib.resources import files


def prompt_loader(package: str) -> Callable[[str], str]:
    """A loader of the prompt templates shipped in one package.

    Args:
        package: Package holding the prompts as ``<name>.md``.

    Returns:
        ``load(name) -> template`` (``{placeholders}`` for LangChain's prompt templates).
    """

    def load(name: str) -> str:
        return files(package).joinpath(f"{name}.md").read_text()

    return load


load_prompt = prompt_loader(__name__)
"""Read a prompt of the shared steps (claim extraction, fact-check judge and rewrite)."""
