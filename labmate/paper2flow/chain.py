"""paper2flow as a LangChain chain: a paper in, ``overview.pdf`` out.

The path through a paper is known in advance, so paper2flow is a chain, not an agent::

    analyze    = ingest | publication | route | extract | outline | write | factcheck | flows
    paper2flow = analyze | render

Every step is a named Runnable (:func:`~labmate.core.lc.step`) that fills in one field of
:class:`~labmate.paper2flow.schemas.Analysis` and saves it as a JSON artifact in the run
directory. The agentic patterns live *inside* the steps: routing, parallel extraction, an
orchestrator (outline), prompt chaining (write), an evaluator-optimizer loop (factcheck)
and orchestrator-workers (flows). paper2post reuses ``analyze``.

A :class:`PaperRun` holds what one run needs: the recorded models and web client, the
tracer and the run directory. Model calls and downloads go through the cassettes, so a
run replays without Ollama or the network.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from labmate.config import Config, ReplayMode
from labmate.core.callbacks import TraceHandler
from labmate.core.extract import extract_claims
from labmate.core.factcheck import fact_check
from labmate.core.ingest import (
    USER_AGENT,
    IngestError,
    download_url,
    first_page_text,
    ingest_arxiv,
    ingest_pdf,
    is_url,
    parse_arxiv_id,
    slugify,
    url_stem,
)
from labmate.core.lc import RecordedChatModel, step
from labmate.core.llm.client import OllamaClient
from labmate.core.llm.replay import CassetteStore, RecordedTransport, ReplayClient, read_lock
from labmate.core.model import LLM
from labmate.core.phases import ModelSwitcher
from labmate.core.traceview import write_trace_html
from labmate.core.tracing import Tracer
from labmate.paper2flow.schemas import Analysis, FactChecked, Outline
from labmate.paper2flow.steps.flow import flow_cards, image_names, plan_flows, render_flows
from labmate.paper2flow.steps.outline import LABELS, plan_outline
from labmate.paper2flow.steps.publication import read_publication, with_publication
from labmate.paper2flow.steps.render import card_blocks, diagrams, main_figure, render_overview
from labmate.paper2flow.steps.route import route_paper
from labmate.paper2flow.steps.write import write_cards

log = logging.getLogger(__name__)

ARTIFACTS = {
    "paper": "00_paper.json",
    "route": "01_route.json",
    "claims": "02_claims.json",
    "outline": "03_outline.json",
    "written": "04_cards.json",
    "checked": "05_factcheck.json",
    "flows": "06_flows.json",
}
"""The JSON artifact each field of :class:`Analysis` is saved to, in step order."""

SourceKind = Literal["arxiv", "url", "pdf"]
"""How a paper is given: an arXiv id or URL, a PDF URL, or a local PDF path."""


class NothingSupportedError(RuntimeError):
    """Raised when the fact-check loop drops every card."""


def need[T](value: T | None, what: str) -> T:
    """Unwrap a field an earlier step must have filled in.

    Args:
        value: The field.
        what: Its name, for the error.

    Returns:
        ``value``.

    Raises:
        ValueError: If the step that fills it in has not run.
    """
    if value is None:
        raise ValueError(f"{what} is missing: the step that produces it has not run")
    return value


def source_kind(source: str) -> SourceKind:
    """Tell how a paper was given.

    Args:
        source: An arXiv id or URL, a PDF URL, or a local PDF path.

    Returns:
        ``"arxiv"``, ``"url"`` or ``"pdf"``.
    """
    if not is_url(source) and source.lower().endswith(".pdf"):
        return "pdf"
    if not is_url(source) or "arxiv.org" in source:
        try:
            parse_arxiv_id(source)
            return "arxiv"
        except IngestError:
            pass
    return "url" if is_url(source) else "pdf"


def run_id_for(source: str) -> str:
    """Directory name of a paper's runs: the arXiv id, or the slug of the file name.

    Args:
        source: The paper (see :func:`source_kind`).

    Returns:
        A filesystem-safe identifier.
    """
    kind = source_kind(source)
    if kind == "arxiv":
        return parse_arxiv_id(source)
    return slugify(url_stem(source) if kind == "url" else Path(source).stem)


def card_labels(outline: Outline, checked: FactChecked) -> list[str]:
    """Labels ("Task", ...) of the cards that survived the fact-check.

    Args:
        outline: The outline (one purpose per planned card).
        checked: Fact-check output (knows which cards were dropped).

    Returns:
        One label per final card.
    """
    dropped = set(checked.report.dropped_cards)
    return [LABELS[c.purpose] for i, c in enumerate(outline.cards, start=1) if i not in dropped]


def method_bullets(outline: Outline, checked: FactChecked) -> list[str]:
    """The fact-checked bullets of the Proposed method card (empty if it was dropped).

    Args:
        outline: The outline.
        checked: Fact-check output.

    Returns:
        Bullet texts.
    """
    labels = card_labels(outline, checked)
    for card, label in zip(checked.cards.cards, labels, strict=True):
        if label == LABELS["method"]:
            return [b.text for b in card.bullets]
    return []


@dataclass
class PaperRun:
    """Everything one run of a paper chain needs.

    Attributes:
        config: Loaded configuration.
        run_dir: ``runs/<paper id>``: the PDF, figures, artifacts, diagrams and trace.
        mode: Replay mode in effect.
        writer: Writer / planner model.
        judge: Critic model (the fact-check judge).
        switcher: Keeps one large model in memory at a time.
        http: Recorded web client (arXiv, PDF downloads).
        tracer: Writes ``trace.jsonl``.
        own_client: The Ollama client the run opened itself (closed with it).
    """

    config: Config
    run_dir: Path
    mode: ReplayMode
    writer: RecordedChatModel
    judge: RecordedChatModel
    switcher: ModelSwitcher
    http: httpx.Client
    tracer: Tracer
    own_client: OllamaClient | None = None

    @property
    def workers(self) -> int:
        """Concurrent model calls within a step."""
        return self.config.pipeline.workers

    def save[M: BaseModel](self, field: str, artifact: M) -> M:
        """Write a field of :class:`Analysis` as its JSON artifact.

        Args:
            field: Field name (a key of :data:`ARTIFACTS`).
            artifact: Its value.

        Returns:
            ``artifact``.
        """
        (self.run_dir / ARTIFACTS[field]).write_text(artifact.model_dump_json(indent=2) + "\n")
        return artifact

    def invoke[I, O](self, chain: Runnable[I, O], value: I, name: str) -> O:
        """Run a chain with labmate's tracing (and LangSmith's, when it is switched on).

        Args:
            chain: The chain.
            value: Its input.
            name: Run name, e.g. ``"paper2flow"``.

        Returns:
            The chain's output.
        """
        handler = TraceHandler(self.tracer, {"paper": self.run_dir.name, "mode": self.mode.value})
        config: RunnableConfig = {"callbacks": [handler], "run_name": name}
        return chain.invoke(value, config)

    def close(self) -> None:
        """Release the model, then close the web client and the run's own Ollama client."""
        self.switcher.release()
        self.http.close()
        if self.own_client is not None:
            self.own_client.close()


def open_run(
    config: Config,
    source: str,
    mode: ReplayMode | None = None,
    client: OllamaClient | None = None,
    transport: httpx.BaseTransport | None = None,
) -> PaperRun:
    """Create the run directory, the recorded models and web client, and the tracer.

    Args:
        config: Loaded configuration.
        source: The paper (see :func:`source_kind`).
        mode: Replay mode override.
        client: Ollama client (built from the config if omitted; unused in replay mode).
        transport: Live web transport (the network if omitted; unused in replay mode).

    Returns:
        The run.
    """
    mode = mode or config.replay.mode
    run_dir = config.tracing.runs_dir / run_id_for(source)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    tracer = Tracer(
        run_dir / "trace.jsonl", trace_id=f"{run_dir.name}@{started}-{uuid.uuid4().hex[:6]}"
    )
    store = CassetteStore(config.replay.dir)
    live, own_client = None, None
    if mode is not ReplayMode.REPLAY:
        if client is None:
            own_client = OllamaClient(config.ollama.host, config.ollama.timeout_s)
        live = client or own_client
        transport = transport or httpx.HTTPTransport(retries=2)
    digests = read_lock(config.replay.lock_file)
    backend = ReplayClient(live, store, mode, digests)
    switcher = ModelSwitcher(live)
    gen, models = config.generation, config.models

    def chat_model(tag: str) -> RecordedChatModel:
        llm = LLM(backend, tag, gen.seed, gen.temperature, gen.num_ctx, digests.get(tag))
        return RecordedChatModel(llm=llm, switcher=switcher)

    http = httpx.Client(
        transport=RecordedTransport(transport, store, mode),
        timeout=120,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    return PaperRun(
        config=config,
        run_dir=run_dir,
        mode=mode,
        writer=chat_model(models.text),
        judge=chat_model(models.critic),
        switcher=switcher,
        http=http,
        tracer=tracer,
        own_client=own_client,
    )


def build_analyze(run: PaperRun) -> Runnable[Analysis, Analysis]:
    """The analysis chain: from a paper reference to fact-checked cards and flow diagrams.

    Args:
        run: The run.

    Returns:
        ``Analysis(source=...) -> Analysis`` with every field filled in.
    """

    def update(a: Analysis, field: str, value: BaseModel) -> Analysis:
        return a.model_copy(update={field: run.save(field, value)})

    def ingest(a: Analysis, config: RunnableConfig) -> Analysis:
        kind = source_kind(a.source)
        if kind == "arxiv":
            paper = ingest_arxiv(a.source, run.run_dir, run.http)
        elif kind == "url":
            downloads = run.config.tracing.runs_dir / ".downloads"
            pdf = download_url(a.source, downloads, run.http)
            paper = ingest_pdf(pdf, title=a.title, url=a.source, run_dir=run.run_dir)
        else:
            paper = ingest_pdf(Path(a.source), title=a.title, run_dir=run.run_dir)
        return update(a, "paper", paper)

    def publication(a: Analysis, config: RunnableConfig) -> Analysis:
        paper = need(a.paper, "paper")
        first_page = first_page_text(run.run_dir / "paper.pdf")
        draft = read_publication(paper, first_page, run.writer, config)
        return update(a, "paper", with_publication(paper, draft))

    def route(a: Analysis, config: RunnableConfig) -> Analysis:
        return update(a, "route", route_paper(need(a.paper, "paper"), run.writer, config))

    def extract(a: Analysis, config: RunnableConfig) -> Analysis:
        claims = extract_claims(need(a.paper, "paper"), run.writer, run.workers, config)
        return update(a, "claims", claims)

    def outline(a: Analysis, config: RunnableConfig) -> Analysis:
        paper, claims = need(a.paper, "paper"), need(a.claims, "claims")
        planned = plan_outline(paper.title, need(a.route, "route"), claims, run.writer, config)
        return update(a, "outline", planned)

    def write(a: Analysis, config: RunnableConfig) -> Analysis:
        outline, claims = need(a.outline, "outline"), need(a.claims, "claims")
        return update(a, "written", write_cards(outline, claims, run.writer, run.workers, config))

    def factcheck(a: Analysis, config: RunnableConfig) -> Analysis:
        outline = need(a.outline, "outline")
        checked = fact_check(
            need(a.written, "written"),
            [c.claim_ids for c in outline.cards],
            need(a.claims, "claims"),
            run.writer,
            run.judge,
            run.config.pipeline.max_rewrite_rounds,
            run.workers,
            need(a.paper, "paper").title,
            config,
        )
        a = update(a, "checked", checked)
        if not checked.cards.cards:
            raise NothingSupportedError(
                f"the fact-check dropped every card; see {run.run_dir / ARTIFACTS['checked']}"
            )
        return a

    def flows(a: Analysis, config: RunnableConfig) -> Analysis:
        outline, checked = need(a.outline, "outline"), need(a.checked, "checked")
        planned = plan_flows(
            need(a.paper, "paper"),
            need(a.route, "route").paper_type,
            method_bullets(outline, checked),
            flow_cards(outline, need(a.claims, "claims").cards),
            run.writer,
            run.workers,
            config,
        )
        render_flows(planned, run.run_dir)
        return update(a, "flows", planned)

    def summary(**fields: Callable[[Analysis], Any]) -> Callable[[Analysis], dict[str, Any]]:
        return lambda a: {name: get(a) for name, get in fields.items()}

    def report(a: Analysis) -> Any:
        return need(a.checked, "checked").report

    return (
        step("ingest", ingest, summary(
            sections=lambda a: len(need(a.paper, "paper").sections),
            figures=lambda a: len(need(a.paper, "paper").figures)))
        | step("publication", publication, summary(
            venue=lambda a: need(a.paper, "paper").venue,
            date=lambda a: need(a.paper, "paper").date))
        | step("route", route, summary(
            paper_type=lambda a: need(a.route, "route").paper_type,
            confidence=lambda a: need(a.route, "route").confidence))
        | step("extract", extract, summary(
            cards=lambda a: len(need(a.claims, "claims").cards),
            rejected=lambda a: len(need(a.claims, "claims").rejected)))
        | step("outline", outline, summary(
            cards=lambda a: len(need(a.outline, "outline").cards)))
        | step("write", write, summary(
            cards=lambda a: len(need(a.written, "written").cards)))
        | step("factcheck", factcheck, summary(
            rounds=lambda a: len(report(a).rounds),
            failed_first=lambda a: report(a).failed_first,
            total_first=lambda a: report(a).total_first,
            dropped=lambda a: len(report(a).dropped)))
        | step("flows", flows, summary(
            nodes=lambda a: len(need(a.flows, "flows").overview.nodes),
            details=lambda a: len(need(a.flows, "flows").details)))
    )  # fmt: skip


def build_paper2flow(run: PaperRun) -> Runnable[Analysis, Path]:
    """The paper2flow chain: :func:`build_analyze`, then the overview is rendered.

    Args:
        run: The run.

    Returns:
        ``Analysis(source=...) -> overview.pdf path``.
    """

    def render(a: Analysis, config: RunnableConfig) -> Path:
        outline, checked = need(a.outline, "outline"), need(a.checked, "checked")
        flows, paper = need(a.flows, "flows"), need(a.paper, "paper")
        labels = card_labels(outline, checked)
        pages = diagrams(flows, image_names(flows), need(a.route, "route").paper_type, run.run_dir)
        return render_overview(
            paper,
            card_blocks(checked.cards, labels),
            pages,
            run.run_dir / "overview.pdf",
            main_figure(checked.cards, labels, paper),
        )

    return build_analyze(run) | step("render", render)


def paper2flow(
    config: Config,
    source: str,
    title: str | None = None,
    mode: ReplayMode | None = None,
    client: OllamaClient | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Turn a paper into ``overview.pdf`` (and write the HTML trace viewer next to it).

    Args:
        config: Loaded configuration.
        source: An arXiv id or URL, a PDF URL, or a local PDF path.
        title: Title override for PDFs without a usable title.
        mode: Replay mode override.
        client: Ollama client (built from the config if omitted).
        transport: Live web transport (the network if omitted).

    Returns:
        Path of ``overview.pdf``.
    """
    run = open_run(config, source, mode, client, transport)
    try:
        overview = run.invoke(
            build_paper2flow(run), Analysis(source=source, title=title), "paper2flow"
        )
    finally:
        run.close()
    write_trace_html(run.run_dir)
    log.info("overview: %s", overview)
    return overview
