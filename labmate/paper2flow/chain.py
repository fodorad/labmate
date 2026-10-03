"""paper2flow as a LangChain chain: a paper in, ``overview.pdf`` out.

The path through a paper is known in advance, so paper2flow is a chain, not an agent::

    analyze    = ingest | extract | write | factcheck | flows
    paper2flow = analyze | render

Every step is a named Runnable (:func:`~labmate.core.lc.step`) that fills in one field of
:class:`~labmate.paper2flow.schemas.Analysis` and saves it as a JSON artifact in the run
directory. The patterns live *inside* the steps: parallel extraction, prompt chaining (write),
an evaluator-optimizer loop (factcheck) and orchestrator-workers (flows). paper2post reuses
``analyze``.

A :class:`PaperRun` holds what one run needs: the models, the web client, the tracer and
the run directory. Model replies are cached (see :mod:`labmate.core.chat`), so rerunning
a paper only calls the model for what changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from labmate.config import Config
from labmate.core.chat import chat_model
from labmate.core.extract import extract_claims
from labmate.core.factcheck import fact_check
from labmate.core.ingest import (
    USER_AGENT,
    IngestError,
    download_url,
    ingest_arxiv,
    ingest_pdf,
    is_url,
    parse_arxiv_id,
    slugify,
    url_stem,
)
from labmate.core.lc import step
from labmate.paper2flow.schemas import Analysis, FactChecked, Outline
from labmate.paper2flow.steps.flow import flow_cards, image_names, plan_flows, render_flows
from labmate.paper2flow.steps.render import card_blocks, diagrams, main_figure, render_overview
from labmate.paper2flow.steps.write import LABELS, plan_cards, write_cards

log = logging.getLogger(__name__)

ARTIFACTS = {
    "paper": "00_paper.json",
    "claims": "01_claims.json",
    "outline": "02_outline.json",
    "written": "03_cards.json",
    "checked": "04_factcheck.json",
    "flows": "05_flows.json",
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
        writer: Writer / planner model.
        judge: Critic model (the fact-check judge).
        http: Web client (arXiv, PDF downloads).
    """

    config: Config
    run_dir: Path
    writer: BaseChatModel
    judge: BaseChatModel
    http: httpx.Client

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
        """Run a chain under a run name (shown in LangSmith or Phoenix when tracing is on).

        Args:
            chain: The chain.
            value: Its input.
            name: Run name, e.g. ``"paper2flow"``.

        Returns:
            The chain's output.
        """
        return chain.invoke(value, {"run_name": name})

    def close(self) -> None:
        """Close the web client."""
        self.http.close()


def open_run(
    config: Config,
    source: str,
    web: httpx.BaseTransport | None = None,
    ollama: httpx.BaseTransport | None = None,
) -> PaperRun:
    """Create the run directory, the models and web client, and the tracer.

    Args:
        config: Loaded configuration.
        source: The paper (see :func:`source_kind`).
        web: Transport for downloads (the network if omitted; tests).
        ollama: Transport to the Ollama server (the configured host if omitted; tests).

    Returns:
        The run.
    """
    run_dir = config.tracing.runs_dir / run_id_for(source)
    run_dir.mkdir(parents=True, exist_ok=True)
    http = httpx.Client(
        transport=web or httpx.HTTPTransport(retries=2),
        timeout=120,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    models = config.models
    return PaperRun(
        config=config,
        run_dir=run_dir,
        writer=chat_model(config, models.text, transport=ollama),
        judge=chat_model(config, models.critic, transport=ollama),
        http=http,
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

    def extract(a: Analysis, config: RunnableConfig) -> Analysis:
        claims = extract_claims(need(a.paper, "paper"), run.writer, run.workers, config)
        return update(a, "claims", claims)

    def write(a: Analysis, config: RunnableConfig) -> Analysis:
        claims = need(a.claims, "claims")
        outline = plan_cards(claims)
        written = write_cards(outline, claims, run.writer, run.workers, config)
        return update(update(a, "outline", outline), "written", written)

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
            method_bullets(outline, checked),
            flow_cards(outline, need(a.claims, "claims").cards),
            run.writer,
            run.workers,
            config,
        )
        render_flows(planned, run.run_dir)
        return update(a, "flows", planned)

    return (
        step("ingest", ingest)
        | step("extract", extract)
        | step("write", write)
        | step("factcheck", factcheck)
        | step("flows", flows)
    )


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
        pages = diagrams(flows, image_names(flows), run.run_dir)
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
    web: httpx.BaseTransport | None = None,
    ollama: httpx.BaseTransport | None = None,
) -> Path:
    """Turn a paper into ``overview.pdf``.

    Args:
        config: Loaded configuration.
        source: An arXiv id or URL, a PDF URL, or a local PDF path.
        title: Title override for PDFs without a usable title.
        web: Transport for downloads (the network if omitted; tests).
        ollama: Transport to the Ollama server (the configured host if omitted; tests).

    Returns:
        Path of ``overview.pdf``.
    """
    run = open_run(config, source, web, ollama)
    try:
        overview = run.invoke(
            build_paper2flow(run), Analysis(source=source, title=title), "paper2flow"
        )
    finally:
        run.close()
    log.info("overview: %s", overview)
    return overview
