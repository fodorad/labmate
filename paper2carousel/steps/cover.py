"""Cover image from the local image model (optional, last memory phase).

Image generation is experimental in Ollama, so failure is not fatal: the carousel simply
gets a text-only cover and the reason is logged and traced.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from paper2carousel.llm.client import Backend, OllamaError
from paper2carousel.llm.types import ImageRequest

log = logging.getLogger(__name__)

COVER_SIZE = (1024, 640)
"""Width x height of the generated image (it fills the top of the 4:5 cover)."""

PROMPT = (
    "Minimalist flat editorial illustration, a visual metaphor for {subject}. Flat geometric "
    "shapes, terracotta orange and slate blue on a warm off-white background, generous "
    "negative space. Purely pictorial: the image contains no writing, no letters, no words, "
    "no numbers, no logos."
)
"""Prompt template; the palette matches the slide theme.

The subject is a lower-case description, never the paper title: image models render a
title they are given as (misspelled) text on the image."""


def cover_subject(title: str, task_title: str | None = None) -> str:
    """What the cover should depict: the task slide's headline, else the title's topic.

    Names (``BlinkLinMulT``, ``RRSI``) and a ``Name:`` prefix are removed, and the result
    is lower-cased, so the image model has nothing it could spell out.

    Args:
        title: Paper title.
        task_title: Headline of the carousel's task slide, if any.

    Returns:
        A short lower-case description.
    """
    from paper2carousel.steps.factcheck import names_in

    text = task_title or title.split(":")[-1]
    names = names_in(text)
    words = [w for w in re.split(r"\s+", text) if w.strip(".,;:!?()").lower() not in names]
    return " ".join(words).strip(" :.,").lower() or "machine learning research"


def make_cover(
    subject: str, backend: Backend, model: str, out: Path, seed: int = 42
) -> tuple[str | None, str]:
    """Generate the cover image.

    Args:
        subject: What to depict (see :func:`cover_subject`).
        backend: Model backend (image calls are recorded and replayed like chat calls).
        model: Image model tag.
        out: Target PNG path.
        seed: Generation seed.

    Returns:
        (file name relative to ``out.parent`` or ``None``, status message).
    """
    width, height = COVER_SIZE
    request = ImageRequest(
        model=model, prompt=PROMPT.format(subject=subject), width=width, height=height, seed=seed
    )
    try:
        image = backend.generate_image(request)
    except (OllamaError, ValueError) as e:
        log.warning("cover image skipped: %s", e)
        return None, f"skipped: {e}"
    out.write_bytes(image.image_bytes())
    return out.name, "ok"
