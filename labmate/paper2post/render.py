"""``post.pdf``, rendered with Typst. Plain code, no LLM.

The post (4:5 pages, the LinkedIn image format): page 1 is the text to copy, one icon
per sentence, the closing question and the links; page 2 is the end-to-end pipeline
diagram, to attach to the post as its image.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from labmate.core.schemas import Paper
from labmate.core.theme import Theme
from labmate.paper2flow.steps.render import author_line, compile_typst, publication_line
from labmate.paper2post.schemas import IconName, Post

ICONS = files("labmate.paper2post").joinpath("icons")
"""The bundled icon SVGs (Tabler Icons, MIT)."""


def colored_icons(names: list[IconName], color: str, out_dir: Path) -> list[str]:
    """Copy the chosen icons next to the output, drawn in ``color``.

    The SVGs draw with ``currentColor``, which Typst renders black, so the colour is
    written into the copies.

    Args:
        names: Icon names, one per sentence.
        color: Stroke colour (hex).
        out_dir: Output directory (the icons go to ``out_dir/icons``).

    Returns:
        Paths relative to ``out_dir``, one per name.
    """
    (out_dir / "icons").mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        svg = ICONS.joinpath(f"{name}.svg").read_text().replace("currentColor", color)
        (out_dir / "icons" / f"{name}.svg").write_text(svg)
        paths.append(f"icons/{name}.svg")
    return paths


def render_post(
    post: Post,
    paper: Paper,
    pipeline: dict[str, Any],
    out: Path,
    theme: Theme | None = None,
) -> Path:
    """Render ``post.pdf``.

    Args:
        post: The finished post (with icons and links).
        paper: The paper.
        pipeline: The end-to-end diagram page (from
            :func:`~labmate.paper2flow.steps.render.diagrams`).
        out: Output PDF path; image paths are relative to its directory.
        theme: Colours and font.

    Returns:
        ``out``.
    """
    theme = theme or Theme()
    icons = colored_icons(post.icons, theme.accent, out.parent)
    data = {
        "hook": post.hook,
        "takeaways": [
            {"text": t.text, "icon": icon}
            for t, icon in zip(post.takeaways, icons or [""] * len(post.takeaways), strict=True)
        ],
        "question": post.question,
        "links": [link.model_dump() for link in post.links],
        "title": paper.title,
        "authors": author_line(paper.authors),
        "publication": publication_line(paper),
        "pipeline": pipeline,
        "theme": theme.model_dump(),
    }
    return compile_typst("labmate.paper2post.templates", "post.typ", data, out)
