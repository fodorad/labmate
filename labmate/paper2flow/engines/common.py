"""What both engines share: the run session and the pipeline stages.

A *stage* is one pipeline step plus its bookkeeping: a trace span, the model phase
(:class:`~labmate.core.phases.ModelSwitcher`), the JSON artifact in the run directory
and a log line. The plain engine calls the stages in order; the LangGraph engine calls
the same stages from graph nodes and replaces the parallel steps with ``Send`` fan-outs
over the same per-item step functions. Identical requests in, identical artifacts out,
which the equivalence test checks in replay mode.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from labmate.config import Config, ReplayMode
from labmate.core.extract import extract_claims
from labmate.core.factcheck import fact_check
from labmate.core.figures import best_figure, short_caption
from labmate.core.ingest import (
    USER_AGENT,
    first_page_text,
    ingest_arxiv,
    ingest_pdf,
    parse_arxiv_id,
    slugify,
)
from labmate.core.llm.client import OllamaClient
from labmate.core.llm.replay import CassetteStore, ReplayClient, read_lock
from labmate.core.model import LLM
from labmate.core.parallel import parallel_map
from labmate.core.phases import ModelSwitcher
from labmate.core.tracing import TracedClient, Tracer
from labmate.paper2flow.schemas import (
    ClaimCard,
    Claims,
    FactChecked,
    FlowDetail,
    FlowOverview,
    Flows,
    Outline,
    Paper,
    Post,
    PublicationDraft,
    Route,
    WrittenSlides,
)
from labmate.paper2flow.steps.flow import (
    assemble_flows,
    detail_order,
    flow_cards,
    flow_sections,
    image_names,
    plan_detail,
    plan_overview,
    render_flows,
)
from labmate.paper2flow.steps.outline import LABELS, plan_outline
from labmate.paper2flow.steps.post import write_post
from labmate.paper2flow.steps.publication import read_publication, with_publication
from labmate.paper2flow.steps.render import blocks, diagrams, render_overview, render_post
from labmate.paper2flow.steps.route import route_paper
from labmate.paper2flow.steps.write import write_slides

log = logging.getLogger("labmate.paper2flow.engines")

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
    overview: Path
    post: Path
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
        run_dir: Where the artifacts go (the same directory as ``paper_dir``).
        trace_id: Id shared by the spans of this invocation.
        mode: Replay mode in effect.
        tracer: Span recorder writing ``trace.jsonl``.
        live: Live Ollama client (``None`` in replay mode).
        llm: Writer / planner settings.
        judge: Critic settings.
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
            overview=self.path("overview.pdf"),
            post=self.path("post.pdf"),
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
    client: OllamaClient | None = None,
) -> Session:
    """Create the run directory, tracer, replaying backend and model settings.

    Args:
        config: Loaded configuration.
        ref: arXiv reference.
        pdf: Local PDF.
        mode: Replay mode override.
        reuse: Reuse existing step artifacts.
        client: Ollama client (built from config if omitted; unused in replay mode).

    Returns:
        The session.
    """
    mode = mode or config.replay.mode
    paper_dir = config.tracing.runs_dir / run_id_for(ref, pdf)
    run_dir = paper_dir
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
        url: Link shown in the outputs for local PDFs (e.g. where it was downloaded from).

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


def stage_publication(s: Session, paper: Paper) -> Paper:
    """Read the venue and publication date off the first page (checked against it).

    Args:
        s: Session.
        paper: The ingested paper.

    Returns:
        The paper with ``venue``, ``date`` and ``year`` filled in.
    """
    s.switcher.use(s.llm.model)
    with s.tracer.span("step.publication") as span:
        pdf = s.paper_dir / "paper.pdf"
        draft = s.checkpoint(
            "00_publication.json",
            PublicationDraft,
            lambda: read_publication(paper, first_page_text(pdf) if pdf.exists() else "", s.llm),
        )
        paper = with_publication(paper, draft)
        span.update(venue=paper.venue, date=paper.date, year=paper.year)
    log.info("published: %s, %s", paper.venue or "(no venue)", paper.date or paper.year)
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
    """Plan the four blocks and assign claims (orchestrator).

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
    log.info("outline: %d blocks, hook: %s", len(draft.slides), draft.hook)
    return draft


def stage_write(s: Session, outline: Outline, claims: Claims) -> WrittenSlides:
    """Write every block from its claims, in parallel (prompt chaining).

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
    log.info("written: %d blocks", len(written.slides))
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
    s: Session, written: WrittenSlides, outline: Outline, claims: Claims, paper: Paper
) -> FactChecked:
    """Check every bullet against its evidence, rewrite or drop failures.

    Args:
        s: Session.
        written: Written slides.
        outline: Approved outline (rewrite scope per slide).
        claims: Claims.
        paper: The paper (its title's names need no quote).

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
                paper.title,
            ),
        )
        span.update(**factcheck_attrs(checked, s.switcher.swaps - swaps_before))
    return record_factcheck(s, checked)


def stage_post(s: Session, checked: FactChecked, claims: Claims, paper: Paper) -> Post:
    """Draft and fact-check the LinkedIn post text.

    Args:
        s: Session.
        checked: Fact-checked blocks.
        claims: Claims.
        paper: The paper.

    Returns:
        The post.
    """
    with s.tracer.span("step.post") as span:
        post = s.checkpoint(
            "08_post.json",
            Post,
            lambda: write_post(checked.slides, claims, paper, s.llm, s.judge, s.switcher),
        )
        span.update(takeaways=len(post.takeaways), dropped=len(post.report.dropped))
    log.info("post: %d sentences", len(post.takeaways))
    return post


def slide_labels(outline: Outline, checked: FactChecked) -> list[str]:
    """Block labels ("Task", ...) of the blocks that survived the fact-check.

    Args:
        outline: Approved outline (one purpose per planned block).
        checked: Fact-check output (knows which blocks were dropped).

    Returns:
        One label per final block.
    """
    dropped = set(checked.report.dropped_slides)
    return [
        LABELS.get(slide.purpose, slide.purpose.replace("_", " ").capitalize())
        for i, slide in enumerate(outline.slides, start=1)
        if i not in dropped
    ]


def method_bullets(outline: Outline, checked: FactChecked) -> list[str]:
    """The fact-checked bullets of the Proposed method block (empty if it was dropped).

    Args:
        outline: Approved outline.
        checked: Fact-check output.

    Returns:
        Bullet texts.
    """
    labels = slide_labels(outline, checked)
    for slide, label in zip(checked.slides.slides, labels, strict=True):
        if label == LABELS["method"]:
            return [b.text for b in slide.bullets]
    return []


@dataclass(frozen=True)
class FlowContext:
    """What the flow planner and its workers see: the evidence cards and section text."""

    cards: list[ClaimCard]
    sections: str
    bullets: list[str]


def flow_context(
    outline: Outline, checked: FactChecked, claims: Claims, paper: Paper
) -> FlowContext:
    """Collect the evidence for the flow diagrams.

    Args:
        outline: Approved outline.
        checked: Fact-check output.
        claims: Claims.
        paper: The paper.

    Returns:
        The context.
    """
    cards = flow_cards(outline, claims.cards)
    return FlowContext(cards, flow_sections(paper, cards), method_bullets(outline, checked))


def stage_flow_overview(s: Session, ctx: FlowContext, paper: Paper, route: Route) -> FlowOverview:
    """Plan the end-to-end flow and pick the steps to break down (orchestrator).

    Args:
        s: Session.
        ctx: Evidence.
        paper: The paper.
        route: Route (what the flow shows).

    Returns:
        The overview.
    """
    s.switcher.use(s.llm.model)
    with s.tracer.span("step.flow.overview") as span:
        overview = plan_overview(
            paper, route.paper_type, ctx.bullets, ctx.cards, ctx.sections, s.llm
        )
        span.update(nodes=len(overview.nodes), expand=detail_order(overview))
    return overview


def flow_detail(
    s: Session, ctx: FlowContext, paper: Paper, overview: FlowOverview, node_id: str
) -> FlowDetail:
    """Draw the detail diagram of one step (worker).

    Args:
        s: Session.
        ctx: Evidence.
        paper: The paper.
        overview: The overview.
        node_id: The step to break down.

    Returns:
        The detail diagram.
    """
    with s.tracer.span("step.flow.detail", step=node_id):
        return plan_detail(paper, overview, node_id, ctx.cards, ctx.sections, s.llm)


def record_flows(s: Session, flows: Flows) -> Flows:
    """Save ``09_flows.json``, render the diagram PNGs and log.

    Args:
        s: Session.
        flows: All diagrams.

    Returns:
        ``flows``.
    """
    save(s.path("09_flows.json"), flows)
    render_flows(flows, s.run_dir)
    log.info(
        "flows: overview with %d boxes, %d detail diagram(s)",
        len(flows.overview.nodes),
        len(flows.details),
    )
    return flows


def stage_flows(
    s: Session, outline: Outline, checked: FactChecked, claims: Claims, paper: Paper, route: Route
) -> Flows:
    """The flow diagrams: the overview, then its details in parallel.

    Args:
        s: Session.
        outline: Approved outline.
        checked: Fact-check output.
        claims: Claims.
        paper: The paper.
        route: Route.

    Returns:
        All diagrams.
    """
    ctx = flow_context(outline, checked, claims, paper)

    def compute() -> Flows:
        overview = stage_flow_overview(s, ctx, paper, route)
        details = parallel_map(
            lambda nid: flow_detail(s, ctx, paper, overview, nid),
            detail_order(overview),
            s.workers,
        )
        return assemble_flows(overview, details)

    with s.tracer.span("step.flows") as span:
        flows = s.checkpoint("09_flows.json", Flows, compute)
        span.update(nodes=len(flows.overview.nodes), details=len(flows.details))
    return record_flows(s, flows)


def main_figure(slides: WrittenSlides, labels: list[str], paper: Paper) -> tuple[str, str] | None:
    """The paper figure for page 1: the one matching the method block best, else Figure 1.

    Args:
        slides: Fact-checked blocks.
        labels: Their labels.
        paper: The paper (its figures).

    Returns:
        ``(path relative to the run directory, caption)``, or ``None`` without figures.
    """
    method = next(
        (sl for sl, label in zip(slides.slides, labels, strict=True) if label == LABELS["method"]),
        None,
    )
    text = " ".join([method.title, *(b.text for b in method.bullets)]) if method else ""
    figure = best_figure(text, paper.figures, paper.title) or next(iter(paper.figures), None)
    return (figure.path, short_caption(figure.caption)) if figure else None


def stage_render(
    s: Session,
    paper: Paper,
    route: Route,
    outline: Outline,
    checked: FactChecked,
    post: Post,
    flows: Flows,
) -> RunResult:
    """Render the two outputs, ``overview.pdf`` and ``post.pdf``, and release the models.

    Args:
        s: Session.
        paper: The paper.
        route: Route.
        outline: Approved outline.
        checked: Fact-checked blocks.
        post: The post.
        flows: The diagrams (PNGs already rendered into the run directory).

    Returns:
        The finished run.
    """
    labels = slide_labels(outline, checked)
    with s.tracer.span("step.render") as span:
        pages = diagrams(flows, image_names(flows), route.paper_type, s.run_dir)
        figure = main_figure(checked.slides, labels, paper)
        render_overview(
            paper, blocks(checked.slides, labels), pages, s.path("overview.pdf"), figure
        )
        render_post(post, paper, pages, s.path("post.pdf"))
        span.update(figure=figure[0] if figure else "", diagrams=len(pages))
    log.info("rendered: %s, %s", s.path("overview.pdf"), s.path("post.pdf"))
    s.switcher.release()
    return s.result("done")
