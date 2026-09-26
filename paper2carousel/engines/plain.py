"""Plain-Python engine: runs the steps in order with checkpoints, tracing and replay.

M1 pipeline (walking skeleton)::

    ingest ──▶ draft (1 LLM call) ──▶ render

Each step writes a JSON artifact into the run directory. Re-running skips steps whose
artifact exists (resume), unless ``fresh=True`` recomputes the LLM steps, which in
``replay`` mode proves the run reproduces from cassettes alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel

from paper2carousel.config import Config, ReplayMode
from paper2carousel.llm.client import OllamaClient
from paper2carousel.llm.replay import CassetteStore, ReplayClient, read_lock
from paper2carousel.schemas import Deck, Paper
from paper2carousel.steps.draft import draft_deck
from paper2carousel.steps.ingest import ingest_arxiv, ingest_pdf, parse_arxiv_id, slugify
from paper2carousel.steps.render import render_deck
from paper2carousel.tracing import TracedClient, Tracer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Paths produced by a run."""

    run_dir: Path
    paper: Path
    deck: Path
    carousel: Path
    trace: Path


def run_id_for(ref: str | None, pdf: Path | None) -> str:
    """Directory name for a run: the arXiv id, or the PDF's slugged stem.

    Args:
        ref: arXiv reference, if any.
        pdf: Local PDF, if any.

    Returns:
        A filesystem-safe identifier.

    Raises:
        ValueError: If neither is given.
    """
    if pdf is not None:
        return slugify(pdf.stem)
    if ref is None:
        raise ValueError("give an arXiv reference or a PDF path")
    return parse_arxiv_id(ref)


def _checkpoint[M: BaseModel](
    path: Path, model: type[M], compute: Callable[[], M], reuse: bool = True
) -> M:
    if reuse and path.exists():
        log.info("  reuse %s", path.name)
        return model.model_validate_json(path.read_text())
    result = compute()
    path.write_text(result.model_dump_json(indent=2) + "\n")
    return result


def run(
    config: Config,
    ref: str | None = None,
    pdf: Path | None = None,
    title: str | None = None,
    mode: ReplayMode | None = None,
    fresh: bool = False,
    client: OllamaClient | None = None,
    http: httpx.Client | None = None,
) -> RunResult:
    """Run the pipeline for one paper.

    Args:
        config: Loaded configuration.
        ref: arXiv id, reference or URL.
        pdf: Local PDF instead of arXiv.
        title: Title override for local PDFs.
        mode: Replay mode override (defaults to ``config.replay.mode``).
        fresh: Recompute LLM steps even if their artifacts exist.
        client: Ollama client (built from config if omitted; unused in ``replay`` mode).
        http: HTTP client for arXiv (built if omitted).

    Returns:
        Paths of the run's artifacts.
    """
    mode = mode or config.replay.mode
    run_dir = config.tracing.runs_dir / run_id_for(ref, pdf)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    tracer = Tracer(run_dir / "trace.jsonl", trace_id=f"{run_dir.name}@{started}")

    live = None
    if mode is not ReplayMode.REPLAY:
        live = client or OllamaClient(config.ollama.host, config.ollama.timeout_s)
    backend = TracedClient(
        ReplayClient(
            live, CassetteStore(config.replay.dir), mode, read_lock(config.replay.lock_file)
        ),
        tracer,
    )
    gen = config.generation
    result = RunResult(
        run_dir=run_dir,
        paper=run_dir / "00_paper.json",
        deck=run_dir / "01_deck.json",
        carousel=run_dir / "carousel.pdf",
        trace=tracer.path or run_dir / "trace.jsonl",
    )

    with tracer.span("run", paper=ref or str(pdf), mode=mode.value, fresh=fresh):
        with tracer.span("step.ingest") as s:
            own_http = http is None
            http = http or httpx.Client(timeout=60, follow_redirects=True)
            try:
                paper = _checkpoint(
                    result.paper,
                    Paper,
                    lambda: (
                        ingest_pdf(pdf, title=title)
                        if pdf is not None
                        else ingest_arxiv(str(ref), run_dir, http)
                    ),
                )
            finally:
                if own_http:
                    http.close()
            s.update(sections=len(paper.sections), title=paper.title)
        log.info("ingested: %s (%d sections)", paper.title, len(paper.sections))

        with tracer.span("step.draft", model=config.models.text) as s:
            deck = _checkpoint(
                result.deck,
                Deck,
                lambda: draft_deck(
                    paper,
                    backend,
                    config.models.text,
                    seed=gen.seed,
                    temperature=gen.temperature,
                    num_ctx=gen.num_ctx,
                ),
                reuse=not fresh,
            )
            s.update(slides=len(deck.slides))
        log.info("drafted: %d slides", len(deck.slides))

        with tracer.span("step.render"):
            render_deck(deck, paper, result.carousel)
        log.info("rendered: %s", result.carousel)

    if live is not None:
        live.unload(config.models.text)
    return result
