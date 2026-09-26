"""Plain-Python engine: runs the steps in order with checkpoints, tracing and replay.

Agentic pipeline (default)::

    ingest ─▶ route ─▶ extract ─▶ outline ─▶ ✋ gate ─▶ write ─▶ fact-check
           ─▶ post ─▶ visuals ─▶ cover image ─▶ render ─▶ slide critic ─▶ summary

Models are used in phases (writer/judge, then image, then vision) so that only one large
model is resident at a time.

Baseline (``--baseline``, the M1 walking skeleton, kept for evaluation)::

    ingest ─▶ draft (one call) ─▶ render

Each step writes a JSON artifact into the run directory. Re-running skips steps whose
artifact exists (resume). ``fresh=True`` recomputes the LLM steps, which in ``replay``
mode proves the run reproduces from cassettes alone. The human's approved outline is an
input, not a model output, so ``fresh`` never discards it.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from paper2carousel.config import Config, ReplayMode
from paper2carousel.llm.client import OllamaClient
from paper2carousel.llm.replay import CassetteStore, ReplayClient, read_lock
from paper2carousel.phases import ModelSwitcher
from paper2carousel.schemas import (
    Claims,
    Deck,
    FactChecked,
    Outline,
    Paper,
    Post,
    Review,
    Route,
    Visuals,
    WrittenSlides,
)
from paper2carousel.steps.cover import make_cover
from paper2carousel.steps.critic import review_deck
from paper2carousel.steps.draft import draft_deck
from paper2carousel.steps.extract import extract_claims
from paper2carousel.steps.factcheck import fact_check
from paper2carousel.steps.gate import read_gate, write_gate
from paper2carousel.steps.ingest import ingest_arxiv, ingest_pdf, parse_arxiv_id, slugify
from paper2carousel.steps.llm import LLM
from paper2carousel.steps.outline import TEMPLATES, plan_outline
from paper2carousel.steps.post import post_markdown, write_post
from paper2carousel.steps.render import render_deck
from paper2carousel.steps.route import route_paper
from paper2carousel.steps.summary import summary_markdown
from paper2carousel.steps.visuals import choose_visuals
from paper2carousel.steps.write import write_slides
from paper2carousel.tracing import TracedClient, Tracer

log = logging.getLogger(__name__)


class NothingSupportedError(RuntimeError):
    """Raised when the fact-check loop drops every slide."""


Status = Literal["done", "awaiting_approval"]
"""Outcome of a run: finished, or paused at the human gate."""


@dataclass(frozen=True)
class RunResult:
    """Where a run's artifacts are and whether it finished."""

    status: Status
    trace_id: str
    run_dir: Path
    carousel: Path
    trace: Path
    gate: Path

    def artifact(self, name: str) -> Path:
        """Path of a step artifact inside the run directory.

        Args:
            name: File name, e.g. ``"02_claims.json"``.

        Returns:
            The path (it may not exist).
        """
        return self.run_dir / name


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
    approve: bool = False,
    auto_approve: bool = False,
    baseline: bool = False,
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
        approve: Accept the (possibly edited) ``outline.yaml`` and continue.
        auto_approve: Skip the human gate (batch runs, evaluation).
        baseline: Run the M1 one-shot pipeline instead of the agentic one.
        client: Ollama client (built from config if omitted; unused in ``replay`` mode).
        http: HTTP client for arXiv (built if omitted).

    Returns:
        Status and artifact paths. ``awaiting_approval`` means ``outline.yaml`` is ready
        for review.
    """
    mode = mode or config.replay.mode
    paper_dir = config.tracing.runs_dir / run_id_for(ref, pdf)
    run_dir = paper_dir / "baseline" if baseline else paper_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    trace_id = f"{run_dir.name}@{started}-{uuid.uuid4().hex[:6]}"
    tracer = Tracer(run_dir / "trace.jsonl", trace_id=trace_id)

    live = None
    if mode is not ReplayMode.REPLAY:
        live = client or OllamaClient(config.ollama.host, config.ollama.timeout_s)
    digests = read_lock(config.replay.lock_file)
    backend = TracedClient(
        ReplayClient(live, CassetteStore(config.replay.dir), mode, digests), tracer, digests
    )
    gen = config.generation
    llm = LLM(backend, config.models.text, gen.seed, gen.temperature, gen.num_ctx)
    judge = LLM(backend, config.models.critic, gen.seed, gen.temperature, gen.num_ctx)
    vision = LLM(backend, config.models.vision, gen.seed, gen.temperature, gen.num_ctx)
    switcher = ModelSwitcher(live)
    workers = config.pipeline.workers
    paths = {
        name: run_dir / name
        for name in [
            "01_route.json",
            "02_claims.json",
            "03_outline.draft.json",
            "03_outline.json",
            "04_slides.json",
            "05_factcheck.json",
            "06_visuals.json",
            "07_review.json",
            "08_post.json",
            "01_deck.json",
        ]
    }

    def result(status: Status) -> RunResult:
        return RunResult(
            status=status,
            trace_id=trace_id,
            run_dir=run_dir,
            carousel=run_dir / "carousel.pdf",
            trace=run_dir / "trace.jsonl",
            gate=run_dir / "outline.yaml",
        )

    with tracer.span("run", paper=ref or str(pdf), mode=mode.value, baseline=baseline) as root:
        with tracer.span("step.ingest") as s:
            own_http = http is None
            http = http or httpx.Client(timeout=60, follow_redirects=True)
            try:
                paper = _checkpoint(
                    paper_dir / "00_paper.json",
                    Paper,
                    lambda: (
                        ingest_pdf(pdf, title=title, run_dir=paper_dir)
                        if pdf is not None
                        else ingest_arxiv(str(ref), paper_dir, http)
                    ),
                )
            finally:
                if own_http:
                    http.close()
            s.update(sections=len(paper.sections), figures=len(paper.figures), title=paper.title)
        log.info(
            "ingested: %s (%d sections, %d figures)",
            paper.title,
            len(paper.sections),
            len(paper.figures),
        )

        switcher.use(llm.model)
        if baseline:
            with tracer.span("step.draft", model=llm.model) as s:
                deck = _checkpoint(
                    paths["01_deck.json"],
                    Deck,
                    lambda: draft_deck(
                        paper, backend, llm.model, llm.seed, llm.temperature, llm.num_ctx
                    ),
                    reuse=not fresh,
                )
                s.update(slides=len(deck.slides))
            return _finish(tracer, deck, paper, result("done"), switcher)

        with tracer.span("step.route") as s:
            route = _checkpoint(
                paths["01_route.json"], Route, lambda: route_paper(paper, llm), reuse=not fresh
            )
            s.update(paper_type=route.paper_type, confidence=route.confidence)
        log.info("routed: %s (%.2f) %s", route.paper_type, route.confidence, route.reason)

        with tracer.span("step.extract", workers=workers) as s:
            claims = _checkpoint(
                paths["02_claims.json"],
                Claims,
                lambda: extract_claims(paper, llm, workers),
                reuse=not fresh,
            )
            s.update(cards=len(claims.cards), rejected=len(claims.rejected))
        log.info("claims: %d verified, %d rejected", len(claims.cards), len(claims.rejected))

        with tracer.span("step.outline") as s:
            draft = _checkpoint(
                paths["03_outline.draft.json"],
                Outline,
                lambda: plan_outline(paper.title, route, claims, llm),
                reuse=not fresh,
            )
            s.update(slides=len(draft.slides))
        log.info("outline: %d slides, hook: %s", len(draft.slides), draft.hook)

        with tracer.span("step.gate", approve=approve, auto_approve=auto_approve) as s:
            approved = paths["03_outline.json"]
            gate = result("done").gate
            if not approved.exists():
                if auto_approve:
                    outline = draft
                elif approve:
                    outline = read_gate(gate, claims, route.paper_type)
                else:
                    write_gate(gate, draft, claims, run_dir.name, TEMPLATES[route.paper_type])
                    s.update(status="awaiting_approval")
                    root.update(status="awaiting_approval")
                    log.info("paused for review: %s", gate)
                    return result("awaiting_approval")
                approved.write_text(outline.model_dump_json(indent=2) + "\n")
            outline = Outline.model_validate_json(approved.read_text())
            s.update(status="approved", edited=outline != draft)

        with tracer.span("step.write", workers=workers) as s:
            written = _checkpoint(
                paths["04_slides.json"],
                WrittenSlides,
                lambda: write_slides(outline, claims, llm, workers),
                reuse=not fresh,
            )
            s.update(slides=len(written.slides))
        log.info("written: %d slides", len(written.slides))

        with tracer.span("step.factcheck", judge=judge.model) as s:
            swaps_before = switcher.swaps
            checked = _checkpoint(
                paths["05_factcheck.json"],
                FactChecked,
                lambda: fact_check(
                    written,
                    [slide.claim_ids for slide in outline.slides],
                    claims,
                    llm,
                    judge,
                    switcher,
                    config.pipeline.max_rewrite_rounds,
                    workers,
                ),
                reuse=not fresh,
            )
            report = checked.report
            s.update(
                rounds=len(report.rounds),
                failed_first=report.failed_first,
                total_first=report.total_first,
                dropped=len(report.dropped),
                dropped_slides=report.dropped_slides,
                swaps=switcher.swaps - swaps_before,
            )
        log.info(
            "fact-check: %d/%d bullets failed the first check, %d dropped after %d round(s)",
            report.failed_first,
            report.total_first,
            len(report.dropped),
            len(report.rounds),
        )
        if not checked.slides.slides:
            switcher.release()
            raise NothingSupportedError(
                f"the fact-check dropped every slide; see {paths['05_factcheck.json']}"
            )
        final = checked.slides.slides

        if config.outputs.post:
            with tracer.span("step.post") as s:
                post = _checkpoint(
                    paths["08_post.json"],
                    Post,
                    lambda: write_post(checked.slides, claims, paper, llm, judge, switcher),
                    reuse=not fresh,
                )
                (run_dir / "post.md").write_text(post_markdown(post, paper))
                s.update(takeaways=len(post.takeaways), dropped=len(post.report.dropped))
            log.info("post: %d takeaways -> post.md", len(post.takeaways))

        with tracer.span("step.visuals", enabled=config.visuals.enabled) as s:
            if config.visuals.enabled:
                switcher.use(llm.model)
                visuals = _checkpoint(
                    paths["06_visuals.json"],
                    Visuals,
                    lambda: choose_visuals(final, paper.figures, paper_dir, llm),
                    reuse=not fresh,
                )
            else:
                visuals = Visuals(slides=[None] * len(final))
            chosen = [v.source for v in visuals.slides if v is not None]
            s.update(visuals=chosen, tool_calls=len(visuals.steps))
        log.info("visuals: %s", ", ".join(chosen) or "none")
        deck = checked.slides.to_deck(visuals.slides)

        if config.visuals.cover_image:
            with tracer.span("step.cover", model=config.models.image) as s:
                cover = run_dir / "cover.png"
                status = "reused"
                if fresh or not cover.exists():
                    switcher.use(config.models.image)
                    _, status = make_cover(
                        paper.title, backend, config.models.image, cover, gen.seed
                    )
                if cover.exists():
                    deck = deck.model_copy(update={"cover_image": cover.name})
                s.update(status=status)
            log.info("cover image: %s", status)

        out = result("done")
        with tracer.span("step.render"):
            render_deck(deck, paper, out.carousel)

        if config.visuals.critic:
            with tracer.span("step.critic", model=vision.model) as s:
                switcher.use(vision.model)
                review = _checkpoint(
                    paths["07_review.json"],
                    Review,
                    lambda: review_deck(deck, out.carousel, vision, workers),
                    reuse=not fresh,
                )
                if review.dropped_visuals:
                    slides = list(deck.slides)
                    for i in review.dropped_visuals:
                        slides[i - 1] = slides[i - 1].model_copy(
                            update={"image": None, "image_caption": None}
                        )
                    deck = deck.model_copy(update={"slides": slides})
                    render_deck(deck, paper, out.carousel)
                alt = [{"page": i, "alt_text": r.alt_text} for i, r in enumerate(review.pages, 1)]
                (run_dir / "alt_texts.json").write_text(json.dumps(alt, indent=2) + "\n")
                flagged = [i for i, r in enumerate(review.pages, 1) if r.overflow or not r.legible]
                s.update(dropped_visuals=review.dropped_visuals, flagged_pages=flagged)
            log.info(
                "critic: dropped visuals %s, flagged pages %s", review.dropped_visuals, flagged
            )

        (run_dir / "summary.md").write_text(summary_markdown(checked.slides, claims, paper))
        log.info("rendered: %s", out.carousel)
        switcher.release()
        return out


def _finish(
    tracer: Tracer, deck: Deck, paper: Paper, result: RunResult, switcher: ModelSwitcher
) -> RunResult:
    with tracer.span("step.render"):
        render_deck(deck, paper, result.carousel)
    log.info("rendered: %s", result.carousel)
    switcher.release()
    return result
