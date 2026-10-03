"""paper2post as a LangChain chain: a paper in, ``post.pdf`` out.

It extends paper2flow's analysis chain::

    paper2post = analyze | post | render

``post`` drafts the post from the overview's fact-checked cards and fact-checks its
sentences again, and gives each an icon by its place in the story; ``render`` lays out the
text, the links and the end-to-end pipeline image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from labmate.config import Config
from labmate.core.lc import step
from labmate.paper2flow.chain import PaperRun, build_analyze, need, open_run
from labmate.paper2flow.schemas import Analysis
from labmate.paper2flow.steps.flow import flow_mermaid, legend, render_mermaid
from labmate.paper2flow.steps.render import FLOW_KICKER, natural_size
from labmate.paper2post.post import post_links, write_post
from labmate.paper2post.render import render_post
from labmate.paper2post.schemas import Post

log = logging.getLogger(__name__)

POST_ARTIFACT = "07_post.json"
"""The post's JSON artifact in the run directory."""

POST_IMAGE = "post-flow.png"
"""The pipeline image of the post: the overview flow without the detail markers."""


class PostState(BaseModel):
    """The analysis plus the post being built.

    Attributes:
        analysis: paper2flow's analysis of the paper.
        post: The post so far.
    """

    analysis: Analysis
    post: Post | None = None


def build_paper2post(run: PaperRun) -> Runnable[Analysis, Path]:
    """The paper2post chain: the analysis, then the post and ``post.pdf``.

    Args:
        run: The run.

    Returns:
        ``Analysis(source=...) -> post.pdf path``.
    """

    def save(state: PostState, post: Post) -> PostState:
        (run.run_dir / POST_ARTIFACT).write_text(post.model_dump_json(indent=2) + "\n")
        return state.model_copy(update={"post": post})

    def draft(a: Analysis, config: RunnableConfig) -> PostState:
        paper = need(a.paper, "paper")
        post = write_post(
            need(a.checked, "checked").cards,
            need(a.claims, "claims"),
            paper,
            run.writer,
            run.judge,
            workers=run.workers,
            config=config,
        )
        post = post.model_copy(update={"links": post_links(paper, run.config.post.links)})
        return save(PostState(analysis=a), post)

    def render(state: PostState, config: RunnableConfig) -> Path:
        a = state.analysis
        overview = need(a.flows, "flows").overview
        (image,) = render_mermaid([flow_mermaid(overview)], [run.run_dir / POST_IMAGE])
        width, height = natural_size(image)
        pipeline = {
            "kicker": FLOW_KICKER,
            "title": overview.title,
            "caption": overview.caption,
            "image": POST_IMAGE,
            "width": round(width, 1),
            "height": round(height, 1),
            "legend": legend(overview),
        }
        return render_post(
            need(state.post, "post"), need(a.paper, "paper"), pipeline, run.run_dir / "post.pdf"
        )

    return build_analyze(run) | step("post", draft) | step("render", render)


def paper2post(
    config: Config,
    source: str,
    title: str | None = None,
    web: httpx.BaseTransport | None = None,
    ollama: httpx.BaseTransport | None = None,
) -> Path:
    """Turn a paper into ``post.pdf``.

    Args:
        config: Loaded configuration.
        source: An arXiv id or URL, a PDF URL, or a local PDF path.
        title: Title override for PDFs without a usable title.
        web: Transport for downloads (the network if omitted; tests).
        ollama: Transport to the Ollama server (the configured host if omitted; tests).

    Returns:
        Path of ``post.pdf``.
    """
    run = open_run(config, source, web, ollama)
    try:
        post = run.invoke(build_paper2post(run), Analysis(source=source, title=title), "paper2post")
    finally:
        run.close()
    log.info("post: %s", post)
    return post
