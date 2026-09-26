"""What both engines share: the run session and the pipeline stages.

A *stage* is one pipeline step plus its bookkeeping: a trace span, the model phase
(:class:`~paper2carousel.phases.ModelSwitcher`), the JSON artifact in the run directory
and a log line. The plain engine calls the stages in order; the LangGraph engine calls
the same stages from graph nodes and replaces the parallel steps with ``Send`` fan-outs
over the same per-item step functions. Identical requests in, identical artifacts out,
which the equivalence test checks in replay mode.
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
    SlideText,
    Visuals,
    WrittenSlides,
)
from paper2carousel.steps.cover import make_cover
from paper2carousel.steps.critic import review_deck
from paper2carousel.steps.extract import extract_claims
from paper2carousel.steps.factcheck import fact_check
from paper2carousel.steps.ingest import (
    USER_AGENT,
    ingest_arxiv,
    ingest_pdf,
    parse_arxiv_id,
    slugify,
)
from paper2carousel.steps.llm import LLM
from paper2carousel.steps.outline import plan_outline
from paper2carousel.steps.post import post_markdown, write_post
from paper2carousel.steps.render import render_deck
from paper2carousel.steps.route import route_paper
from paper2carousel.steps.summary import summary_markdown
from paper2carousel.steps.visuals import choose_visuals
from paper2carousel.steps.write import write_slides
from paper2carousel.tracing import TracedClient, Tracer

log = logging.getLogger("paper2carousel.engines")

Status = Literal["done", "awaiting_approval"]
"""Outcome of a run: finished, or paused at the human gate."""


class NothingSupportedError(RuntimeError):
    """Raised when the fact-check loop drops every slide."""


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


@dataclass
class Session:
    """Everything a run needs: directories, tracer, models, phase switcher.

    Attributes:
        config: Loaded configuration.
        paper_dir: ``runs/<paper_id>`` (the PDF, figures and ``00_paper.json`` live here).
        run_dir: Where the artifacts go (``paper_dir``, or ``paper_dir/baseline``).
        trace_id: Id shared by the spans of this invocation.
        mode: Replay mode in effect.
        tracer: Span recorder writing ``trace.jsonl``.
        live: Live Ollama client (``None`` in replay mode).
        llm: Writer / planner settings.
        judge: Critic settings.
        vision: Vision model settings.
        switcher: Keeps one large model resident at a time.
        reuse: Reuse existing step artifacts instead of recomputing them.
    """

    config: Config
    paper_dir: Path
    run_dir: Path
    trace_id: str
    mode: ReplayMode
    tracer: Tracer
    live: OllamaClient | None
    llm: LLM
    judge: LLM
    vision: LLM
    switcher: ModelSwitcher
    reuse: bool

    @property
    def workers(self) -> int:
        """Concurrent model calls within a phase."""
        return self.config.pipeline.workers

    def path(self, name: str) -> Path:
        """Artifact path in the run directory.

        Args:
            name: File name.

        Returns:
            ``run_dir / name``.
        """
        return self.run_dir / name

    def result(self, status: Status) -> RunResult:
        """Run result for this session.

        Args:
            status: Outcome.

        Returns:
            The result with the standard artifact paths.
        """
        return RunResult(
            status=status,
            trace_id=self.trace_id,
            run_dir=self.run_dir,
            carousel=self.path("carousel.pdf"),
            trace=self.path("trace.jsonl"),
            gate=self.path("outline.yaml"),
        )

    def checkpoint[M: BaseModel](
        self, name: str, model: type[M], compute: Callable[[], M], reuse: bool | None = None
    ) -> M:
        """Load an artifact if it exists and reuse is on, else compute and save it.

        Args:
            name: Artifact file name.
            model: Its schema.
            compute: Produces the artifact.
            reuse: Override :attr:`reuse` (``True`` for inputs such as the paper).

        Returns:
            The artifact.
        """
        path = self.path(name) if name != "00_paper.json" else self.paper_dir / name
        if (self.reuse if reuse is None else reuse) and path.exists():
            log.info("  reuse %s", path.name)
            return model.model_validate_json(path.read_text())
        return save(path, compute())


def save[M: BaseModel](path: Path, artifact: M) -> M:
    """Write an artifact as indented JSON (the format both engines produce).

    Args:
        path: Target file.
        artifact: Model to write.

    Returns:
        ``artifact``.
    """
    path.write_text(artifact.model_dump_json(indent=2) + "\n")
    return artifact


def open_session(
    config: Config,
    ref: str | None,
    pdf: Path | None,
    mode: ReplayMode | None,
    reuse: bool,
    baseline: bool = False,
    client: OllamaClient | None = None,
) -> Session:
    """Create the run directory, tracer, replaying backend and model settings.

    Args:
        config: Loaded configuration.
        ref: arXiv reference.
        pdf: Local PDF.
        mode: Replay mode override.
        reuse: Reuse existing step artifacts.
        baseline: Put artifacts under ``baseline/``.
        client: Ollama client (built from config if omitted; unused in replay mode).

    Returns:
        The session.
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
    m = config.models
    return Session(
        config=config,
        paper_dir=paper_dir,
        run_dir=run_dir,
        trace_id=trace_id,
        mode=mode,
        tracer=tracer,
        live=live,
        llm=LLM(backend, m.text, gen.seed, gen.temperature, gen.num_ctx),
        judge=LLM(backend, m.critic, gen.seed, gen.temperature, gen.num_ctx),
        vision=LLM(backend, m.vision, gen.seed, gen.temperature, gen.num_ctx),
        switcher=ModelSwitcher(live),
        reuse=reuse,
    )


# --- stages --------------------------------------------------------------------------------


def stage_ingest(
    s: Session,
    ref: str | None,
    pdf: Path | None,
    title: str | None,
    http: httpx.Client | None,
    url: str = "",
) -> Paper:
    """Fetch and parse the paper (always reused once ingested: it is an input).

    Args:
        s: Session.
        ref: arXiv reference.
        pdf: Local PDF.
        title: Title override for local PDFs.
        http: HTTP client for arXiv (built if omitted).
        url: Link shown on the slides for local PDFs (e.g. where it was downloaded from).

    Returns:
        The paper.
    """
    with s.tracer.span("step.ingest") as span:
        own_http = http is None
        client = http or httpx.Client(
            timeout=60, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        try:
            paper = s.checkpoint(
                "00_paper.json",
                Paper,
                lambda: (
                    ingest_pdf(pdf, title=title, url=url, run_dir=s.paper_dir)
                    if pdf is not None
                    else ingest_arxiv(str(ref), s.paper_dir, client)
                ),
                reuse=True,
            )
        finally:
            if own_http:
                client.close()
        span.update(sections=len(paper.sections), figures=len(paper.figures), title=paper.title)
    log.info(
        "ingested: %s (%d sections, %d figures)",
        paper.title,
        len(paper.sections),
        len(paper.figures),
    )
    return paper


def stage_route(s: Session, paper: Paper) -> Route:
    """Route the paper to a slide template (routing).

    Args:
        s: Session.
        paper: The paper.

    Returns:
        The route.
    """
    s.switcher.use(s.llm.model)
    with s.tracer.span("step.route") as span:
        route = s.checkpoint("01_route.json", Route, lambda: route_paper(paper, s.llm))
        span.update(paper_type=route.paper_type, confidence=route.confidence)
    log.info("routed: %s (%.2f) %s", route.paper_type, route.confidence, route.reason)
    return route


def record_claims(s: Session, claims: Claims) -> None:
    """Log the extraction result.

    Args:
        s: Session.
        claims: Verified and rejected claims.
    """
    log.info("claims: %d verified, %d rejected", len(claims.cards), len(claims.rejected))


def stage_extract(s: Session, paper: Paper) -> Claims:
    """Extract and verify claim cards, sections in parallel (parallelisation).

    Args:
        s: Session.
        paper: The paper.

    Returns:
        The claims.
    """
    with s.tracer.span("step.extract", workers=s.workers) as span:
        claims = s.checkpoint(
            "02_claims.json", Claims, lambda: extract_claims(paper, s.llm, s.workers)
        )
        span.update(cards=len(claims.cards), rejected=len(claims.rejected))
    record_claims(s, claims)
    return claims


def stage_outline(s: Session, paper: Paper, route: Route, claims: Claims) -> Outline:
    """Plan the slides and assign claims (orchestrator).

    Args:
        s: Session.
        paper: The paper.
        route: Its route.
        claims: Verified claims.

    Returns:
        The draft outline (before the human gate).
    """
    with s.tracer.span("step.outline") as span:
        draft = s.checkpoint(
            "03_outline.draft.json",
            Outline,
            lambda: plan_outline(paper.title, route, claims, s.llm),
        )
        span.update(slides=len(draft.slides))
    log.info("outline: %d slides, hook: %s", len(draft.slides), draft.hook)
    return draft


def stage_write(s: Session, outline: Outline, claims: Claims) -> WrittenSlides:
    """Write every slide from its claims, in parallel (prompt chaining).

    Args:
        s: Session.
        outline: Approved outline.
        claims: Claims.

    Returns:
        The written slides.
    """
    with s.tracer.span("step.write", workers=s.workers) as span:
        written = s.checkpoint(
            "04_slides.json",
            WrittenSlides,
            lambda: write_slides(outline, claims, s.llm, s.workers),
        )
        span.update(slides=len(written.slides))
    log.info("written: %d slides", len(written.slides))
    return written


def record_factcheck(s: Session, checked: FactChecked) -> FactChecked:
    """Log the fact-check result and stop if nothing survived.

    Args:
        s: Session.
        checked: Fact-check output.

    Returns:
        ``checked``.

    Raises:
        NothingSupportedError: If every slide was dropped.
    """
    report = checked.report
    log.info(
        "fact-check: %d/%d bullets failed the first check, %d dropped after %d round(s)",
        report.failed_first,
        report.total_first,
        len(report.dropped),
        len(report.rounds),
    )
    if not checked.slides.slides:
        s.switcher.release()
        raise NothingSupportedError(
            f"the fact-check dropped every slide; see {s.path('05_factcheck.json')}"
        )
    return checked


def factcheck_attrs(checked: FactChecked, swaps: int) -> dict[str, object]:
    """Span attributes of the fact-check step.

    Args:
        checked: Fact-check output.
        swaps: Model swaps during the loop.

    Returns:
        Attributes.
    """
    report = checked.report
    return {
        "rounds": len(report.rounds),
        "failed_first": report.failed_first,
        "total_first": report.total_first,
        "dropped": len(report.dropped),
        "dropped_slides": report.dropped_slides,
        "swaps": swaps,
    }


def stage_factcheck(
    s: Session, written: WrittenSlides, outline: Outline, claims: Claims
) -> FactChecked:
    """Check every bullet against its evidence, rewrite or drop failures.

    Args:
        s: Session.
        written: Written slides.
        outline: Approved outline (rewrite scope per slide).
        claims: Claims.

    Returns:
        The corrected slides and the audit.
    """
    with s.tracer.span("step.factcheck", judge=s.judge.model) as span:
        swaps_before = s.switcher.swaps
        checked = s.checkpoint(
            "05_factcheck.json",
            FactChecked,
            lambda: fact_check(
                written,
                [slide.claim_ids for slide in outline.slides],
                claims,
                s.llm,
                s.judge,
                s.switcher,
                s.config.pipeline.max_rewrite_rounds,
                s.workers,
            ),
        )
        span.update(**factcheck_attrs(checked, s.switcher.swaps - swaps_before))
    return record_factcheck(s, checked)


def stage_post(s: Session, checked: FactChecked, claims: Claims, paper: Paper) -> None:
    """Draft and fact-check the LinkedIn post (if enabled).

    Args:
        s: Session.
        checked: Fact-checked slides.
        claims: Claims.
        paper: The paper.
    """
    if not s.config.outputs.post:
        return
    with s.tracer.span("step.post") as span:
        post = s.checkpoint(
            "08_post.json",
            Post,
            lambda: write_post(checked.slides, claims, paper, s.llm, s.judge, s.switcher),
        )
        s.path("post.md").write_text(post_markdown(post, paper))
        span.update(takeaways=len(post.takeaways), dropped=len(post.report.dropped))
    log.info("post: %d takeaways -> post.md", len(post.takeaways))


def stage_visuals(
    s: Session, slides: list[SlideText], paper: Paper, claims: Claims | None = None
) -> Visuals:
    """Let the visuals agent pick a figure or diagram per slide (tool use).

    Args:
        s: Session.
        slides: Final slides.
        paper: The paper (figures).
        claims: Claim cards (evidence for charts).

    Returns:
        One optional visual per slide and the agent's tool calls.
    """
    enabled = s.config.visuals.enabled
    with s.tracer.span("step.visuals", enabled=enabled) as span:
        if enabled:
            s.switcher.use(s.llm.model)
            visuals = s.checkpoint(
                "06_visuals.json",
                Visuals,
                lambda: choose_visuals(
                    slides, paper.figures, s.paper_dir, s.llm, paper.title, claims
                ),
            )
        else:
            visuals = Visuals(slides=[None] * len(slides))
        chosen = [v.source for v in visuals.slides if v is not None]
        span.update(visuals=chosen, tool_calls=len(visuals.steps))
    log.info("visuals: %s", ", ".join(chosen) or "none")
    return visuals


def stage_cover(s: Session, deck: Deck, paper: Paper) -> Deck:
    """Generate the cover illustration (if enabled; failures are not fatal).

    Args:
        s: Session.
        deck: Deck so far.
        paper: The paper.

    Returns:
        The deck, with ``cover_image`` set if a cover exists.
    """
    if not s.config.visuals.cover_image:
        return deck
    image_model = s.config.models.image
    with s.tracer.span("step.cover", model=image_model) as span:
        cover = s.path("cover.png")
        status = "reused"
        if not s.reuse or not cover.exists():
            s.switcher.use(image_model)
            _, status = make_cover(
                paper.title, s.llm.backend, image_model, cover, s.config.generation.seed
            )
        if cover.exists():
            deck = deck.model_copy(update={"cover_image": cover.name})
        span.update(status=status)
    log.info("cover image: %s", status)
    return deck


def stage_render(s: Session, deck: Deck, paper: Paper) -> Path:
    """Render the carousel PDF.

    Args:
        s: Session.
        deck: Deck to render.
        paper: The paper.

    Returns:
        The PDF path.
    """
    out = s.path("carousel.pdf")
    with s.tracer.span("step.render"):
        render_deck(deck, paper, out)
    return out


def stage_critic(s: Session, deck: Deck, paper: Paper) -> Deck:
    """Review the rendered pages with the vision model; drop visuals it rejects.

    Args:
        s: Session.
        deck: The rendered deck.
        paper: The paper.

    Returns:
        The deck as finally rendered.
    """
    if not s.config.visuals.critic:
        return deck
    out = s.path("carousel.pdf")
    with s.tracer.span("step.critic", model=s.vision.model) as span:
        s.switcher.use(s.vision.model)
        review = s.checkpoint(
            "07_review.json", Review, lambda: review_deck(deck, out, s.vision, s.workers)
        )
        if review.dropped_visuals:
            slides = list(deck.slides)
            for i in review.dropped_visuals:
                slides[i - 1] = slides[i - 1].model_copy(
                    update={"image": None, "image_caption": None}
                )
            deck = deck.model_copy(update={"slides": slides})
            render_deck(deck, paper, out)
        alt = [{"page": i, "alt_text": r.alt_text} for i, r in enumerate(review.pages, 1)]
        s.path("alt_texts.json").write_text(json.dumps(alt, indent=2) + "\n")
        flagged = [i for i, r in enumerate(review.pages, 1) if r.overflow or not r.legible]
        span.update(dropped_visuals=review.dropped_visuals, flagged_pages=flagged)
    log.info("critic: dropped visuals %s, flagged pages %s", review.dropped_visuals, flagged)
    return deck


def stage_summary(s: Session, checked: FactChecked, claims: Claims, paper: Paper) -> RunResult:
    """Write ``summary.md``, release the models and report the result.

    Args:
        s: Session.
        checked: Fact-checked slides.
        claims: Claims.
        paper: The paper.

    Returns:
        The finished run.
    """
    s.path("summary.md").write_text(summary_markdown(checked.slides, claims, paper))
    log.info("rendered: %s", s.path("carousel.pdf"))
    s.switcher.release()
    return s.result("done")
