"""Cover image from the local image model (optional, last memory phase).

Image generation is experimental in Ollama, so failure is not fatal: the carousel simply
gets a text-only cover and the reason is logged and traced.
"""

from __future__ import annotations

import logging
from pathlib import Path

from paper2carousel.llm.client import Backend, OllamaError
from paper2carousel.llm.types import ImageRequest

log = logging.getLogger(__name__)

COVER_SIZE = (1024, 640)
"""Width x height of the generated image (it fills the top of the 4:5 cover)."""

PROMPT = (
    "Abstract minimalist editorial illustration representing the idea of: {title}. "
    "Flat geometric shapes, terracotta orange and slate blue on a warm off-white background, "
    "generous negative space, soft light. No text, no letters, no numbers, no logos."
)
"""Prompt template; the palette matches the slide theme."""


def make_cover(
    title: str, backend: Backend, model: str, out: Path, seed: int = 42
) -> tuple[str | None, str]:
    """Generate the cover image.

    Args:
        title: Paper title (the image's subject).
        backend: Model backend (image calls are recorded and replayed like chat calls).
        model: Image model tag.
        out: Target PNG path.
        seed: Generation seed.

    Returns:
        (file name relative to ``out.parent`` or ``None``, status message).
    """
    width, height = COVER_SIZE
    request = ImageRequest(
        model=model, prompt=PROMPT.format(title=title), width=width, height=height, seed=seed
    )
    try:
        image = backend.generate_image(request)
    except (OllamaError, ValueError) as e:
        log.warning("cover image skipped: %s", e)
        return None, f"skipped: {e}"
    out.write_bytes(image.image_bytes())
    return out.name, "ok"
