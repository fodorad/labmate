"""LangGraph engine: the same stages, orchestrated as a ``StateGraph``.

The graph (``Send`` fans out, ``⟲`` is a cycle, ``✋`` an interrupt)::

    ingest ─▶ route ─▶ plan_extract ─Send per section─▶ extract_unit ─▶ assemble_claims
      ─▶ outline ─▶ propose ─▶ ✋ gate ─Send per slide─▶ write_slide ─▶ assemble_slides
      ─▶ judge ⟲ rewrite ─▶ finish_factcheck ─▶ post ─▶ visuals ─▶ cover ─▶ render
      ─▶ critic ─▶ summary

What LangGraph provides here, compared with the plain engine:

- **Fan-out/fan-in** with ``Send`` and a reducer instead of a thread pool.
- **The evaluator–optimizer loop as a graph cycle** (``judge`` ⇄ ``rewrite``) with a
  conditional edge, instead of a ``while`` loop.
- **Pause/resume at the human gate** with ``interrupt()`` and a SQLite checkpointer
  (``runs/<id>/langgraph.sqlite``), instead of re-running with file checkpoints.

The step functions are the ones the plain engine uses, so both engines send identical
model requests; in replay mode they must write byte-identical artifacts (tested in CI).
"""

from __future__ import annotations

import operator
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any, TypedDict

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel

from paper2carousel import schemas
from paper2carousel.config import Config, ReplayMode
from paper2carousel.engines.common import (
    RunResult,
    Session,
    factcheck_attrs,
    log,
    open_session,
    record_claims,
    record_factcheck,
    save,
    stage_cover,
    stage_critic,
    stage_ingest,
    stage_outline,
    stage_post,
    stage_render,
    stage_route,
    stage_summary,
    stage_visuals,
)
from paper2carousel.llm.client import OllamaClient
from paper2carousel.schemas import (
    ClaimDraft,
    Claims,
    Deck,
    FactChecked,
    Outline,
    OutlineSlide,
    Paper,
    Route,
    Section,
    SlideText,
    Visuals,
    WrittenSlides,
)
from paper2carousel.steps.extract import assemble_claims, extract_unit, extraction_units
from paper2carousel.steps.factcheck import (
    FactCheckLoop,
    finish_loop,
    judge_pending,
    rewrite_pending,
    should_rewrite,
    start_loop,
)
from paper2carousel.steps.gate import read_gate, write_gate
from paper2carousel.steps.outline import TEMPLATES
from paper2carousel.steps.write import write_slide

THREAD: RunnableConfig = {"configurable": {"thread_id": "run"}}
"""One thread per run directory (each run has its own checkpoint database)."""


class State(TypedDict, total=False):
    """Graph state. Fan-out results arrive through ``operator.add`` reducers."""

    paper: Paper
    route: Route
    units: list[Section]
    drafts: Annotated[list[tuple[int, list[ClaimDraft]]], operator.add]
    claims: Claims
    draft: Outline
    outline: Outline
    written_parts: Annotated[list[tuple[int, SlideText]], operator.add]
    written: WrittenSlides
    loop: FactCheckLoop
    swaps_before: int
    checked: FactChecked
    visuals: Visuals
    deck: Deck


class UnitTask(TypedDict):
    """Payload of one ``Send`` to ``extract_unit``."""

    index: int
    unit: Section
    title: str


class SlideTask(TypedDict):
    """Payload of one ``Send`` to ``write_slide``."""

    index: int
    planned: OutlineSlide
    total: int
    claims: Claims


def _serializer() -> JsonPlusSerializer:
    """Checkpoint serializer that may rebuild this package's pydantic models (and no others)."""
    models = [
        v
        for v in vars(schemas).values()
        if isinstance(v, type) and issubclass(v, BaseModel) and v.__module__ == schemas.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=[*models, FactCheckLoop])


def build_graph(
    s: Session,
    ref: str | None,
    pdf: Path | None,
    title: str | None,
    http: httpx.Client | None,
    url: str,
    approve: bool,
    auto_approve: bool,
) -> StateGraph:
    """Wire the stages into a graph (not compiled).

    Args:
        s: Session (closed over by the nodes; not part of the checkpointed state).
        ref: arXiv reference.
        pdf: Local PDF.
        title: Title override for local PDFs.
        http: HTTP client for arXiv.
        url: Link to a local PDF's source.
        approve: Accept ``outline.yaml`` without pausing.
        auto_approve: Accept the draft outline without review.

    Returns:
        The graph builder.
    """

    def ingest(state: State) -> State:
        return {"paper": stage_ingest(s, ref, pdf, title, http, url)}

    def route(state: State) -> State:
        return {"route": stage_route(s, state["paper"])}

    def plan_extract(state: State) -> State:
        return {"units": extraction_units(state["paper"])}

    def fan_out_units(state: State) -> list[Send]:
        title = state["paper"].title
        return [
            Send("extract_unit", UnitTask(index=i, unit=u, title=title))
            for i, u in enumerate(state["units"])
        ]

    def extract_one(task: UnitTask) -> State:
        with s.tracer.span("step.extract.unit", section=task["unit"].title):
            return {"drafts": [(task["index"], extract_unit(task["title"], task["unit"], s.llm))]}

    def assemble(state: State) -> State:
        with s.tracer.span("step.extract", workers=s.workers) as span:
            drafts = [d for _, d in sorted(state.get("drafts", []), key=lambda x: x[0])]
            claims = save(s.path("02_claims.json"), assemble_claims(state["units"], drafts))
            span.update(cards=len(claims.cards), rejected=len(claims.rejected))
        record_claims(s, claims)
        return {"claims": claims}

    def outline(state: State) -> State:
        return {"draft": stage_outline(s, state["paper"], state["route"], state["claims"])}

    def propose(state: State) -> State:
        if not (s.path("03_outline.json").exists() or approve or auto_approve):
            route_ = state["route"]
            write_gate(
                s.path("outline.yaml"),
                state["draft"],
                state["claims"],
                s.run_dir.name,
                TEMPLATES[route_.paper_type],
            )
            log.info("paused for review: %s", s.path("outline.yaml"))
        return {}

    def gate(state: State) -> State:
        approved = s.path("03_outline.json")
        if not (approved.exists() or approve or auto_approve):
            interrupt({"review": str(s.path("outline.yaml"))})  # pauses; resumes here
        with s.tracer.span("step.gate", approve=approve, auto_approve=auto_approve) as span:
            if not approved.exists():
                chosen = (
                    state["draft"]
                    if auto_approve
                    else read_gate(
                        s.path("outline.yaml"), state["claims"], state["route"].paper_type
                    )
                )
                approved.write_text(chosen.model_dump_json(indent=2) + "\n")
            result = Outline.model_validate_json(approved.read_text())
            span.update(status="approved", edited=result != state["draft"])
        s.switcher.use(s.llm.model)
        return {"outline": result}

    def fan_out_slides(state: State) -> list[Send]:
        slides = state["outline"].slides
        return [
            Send(
                "write_slide",
                SlideTask(index=i, planned=p, total=len(slides), claims=state["claims"]),
            )
            for i, p in enumerate(slides)
        ]

    def write_one(task: SlideTask) -> State:
        by_id = {c.id: c for c in task["claims"].cards}
        with s.tracer.span("step.write.slide", slide=task["index"] + 1):
            slide = write_slide(task["index"] + 1, task["planned"], task["total"], by_id, s.llm)
        return {"written_parts": [(task["index"], slide)]}

    def assemble_slides(state: State) -> State:
        with s.tracer.span("step.write", workers=s.workers) as span:
            parts = sorted(state.get("written_parts", []), key=lambda x: x[0])
            written = WrittenSlides(hook=state["outline"].hook, slides=[p for _, p in parts])
            save(s.path("04_slides.json"), written)
            span.update(slides=len(written.slides))
        log.info("written: %d slides", len(written.slides))
        return {"written": written, "loop": start_loop(written), "swaps_before": s.switcher.swaps}

    def judge(state: State) -> State:
        by_id = {c.id: c for c in state["claims"].cards}
        s.switcher.use(s.judge.model)
        with s.tracer.span("step.factcheck.judge", round=len(state["loop"].report.rounds)):
            return {"loop": judge_pending(state["loop"], by_id, s.judge, s.workers)}

    def after_judge(state: State) -> str:
        rounds = s.config.pipeline.max_rewrite_rounds
        return "rewrite" if should_rewrite(state["loop"], rounds) else "finish_factcheck"

    def rewrite(state: State) -> State:
        by_id = {c.id: c for c in state["claims"].cards}
        scope = [slide.claim_ids for slide in state["outline"].slides]
        s.switcher.use(s.llm.model)
        with s.tracer.span("step.factcheck.rewrite", slides=len(state["loop"].pending)):
            return {"loop": rewrite_pending(state["loop"], scope, by_id, s.llm, s.workers)}

    def finish_factcheck(state: State) -> State:
        with s.tracer.span("step.factcheck", judge=s.judge.model) as span:
            checked = save(s.path("05_factcheck.json"), finish_loop(state["loop"]))
            span.update(**factcheck_attrs(checked, s.switcher.swaps - state["swaps_before"]))
        return {"checked": record_factcheck(s, checked)}

    def post(state: State) -> State:
        stage_post(s, state["checked"], state["claims"], state["paper"])
        return {}

    def visuals(state: State) -> State:
        return {"visuals": stage_visuals(s, state["checked"].slides.slides, state["paper"])}

    def cover(state: State) -> State:
        deck = state["checked"].slides.to_deck(state["visuals"].slides)
        return {"deck": stage_cover(s, deck, state["paper"])}

    def render(state: State) -> State:
        stage_render(s, state["deck"], state["paper"])
        return {}

    def critic(state: State) -> State:
        return {"deck": stage_critic(s, state["deck"], state["paper"])}

    def summary(state: State) -> State:
        stage_summary(s, state["checked"], state["claims"], state["paper"])
        return {}

    g = StateGraph(State)
    for name, fn in [
        ("ingest", ingest),
        ("route", route),
        ("plan_extract", plan_extract),
        ("extract_unit", extract_one),
        ("assemble_claims", assemble),
        ("outline", outline),
        ("propose", propose),
        ("gate", gate),
        ("write_slide", write_one),
        ("assemble_slides", assemble_slides),
        ("judge", judge),
        ("rewrite", rewrite),
        ("finish_factcheck", finish_factcheck),
        ("post", post),
        ("visuals", visuals),
        ("cover", cover),
        ("render", render),
        ("critic", critic),
        ("summary", summary),
    ]:
        g.add_node(name, fn)  # type: ignore[call-overload]
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "route")
    g.add_edge("route", "plan_extract")
    g.add_conditional_edges("plan_extract", fan_out_units, ["extract_unit"])
    g.add_edge("extract_unit", "assemble_claims")
    g.add_edge("assemble_claims", "outline")
    g.add_edge("outline", "propose")
    g.add_edge("propose", "gate")
    g.add_conditional_edges("gate", fan_out_slides, ["write_slide"])
    g.add_edge("write_slide", "assemble_slides")
    g.add_edge("assemble_slides", "judge")
    g.add_conditional_edges("judge", after_judge, ["rewrite", "finish_factcheck"])
    g.add_edge("rewrite", "judge")
    for a, b in [
        ("finish_factcheck", "post"),
        ("post", "visuals"),
        ("visuals", "cover"),
        ("cover", "render"),
        ("render", "critic"),
        ("critic", "summary"),
        ("summary", END),
    ]:
        g.add_edge(a, b)
    return g


def run(
    config: Config,
    ref: str | None = None,
    pdf: Path | None = None,
    title: str | None = None,
    url: str = "",
    mode: ReplayMode | None = None,
    fresh: bool = False,
    approve: bool = False,
    auto_approve: bool = False,
    client: OllamaClient | None = None,
    http: httpx.Client | None = None,
) -> RunResult:
    """Run the pipeline for one paper with the LangGraph engine.

    Model steps are always recomputed (cassettes make that cheap); resuming after the
    human gate uses the graph's checkpoint instead of reloading step artifacts.

    Args:
        config: Loaded configuration.
        ref: arXiv id, reference or URL.
        pdf: Local PDF instead of arXiv.
        title: Title override for local PDFs.
        url: Link to a local PDF's source (shown on the slides).
        mode: Replay mode override.
        fresh: Discard the graph checkpoint and start over.
        approve: Resume a paused run (or accept ``outline.yaml`` without pausing).
        auto_approve: Skip the human gate.
        client: Ollama client (unused in replay mode).
        http: HTTP client for arXiv.

    Returns:
        Status and artifact paths, as in the plain engine.
    """
    s = open_session(config, ref, pdf, mode, reuse=False, client=client)
    db = s.run_dir / "langgraph.sqlite"
    if fresh and db.exists():
        db.unlink()
    root_attrs = {"paper": ref or str(pdf), "mode": s.mode.value, "baseline": False}
    with (
        s.tracer.span("run", engine="langgraph", **root_attrs) as root,
        closing(sqlite3.connect(db, check_same_thread=False)) as conn,
    ):
        saver = SqliteSaver(conn, serde=_serializer())
        graph = build_graph(s, ref, pdf, title, http, url, approve, auto_approve).compile(
            checkpointer=saver
        )
        paused = bool(graph.get_state(THREAD).interrupts)
        inputs: Any = Command(resume=True) if approve and paused else {}
        result = graph.invoke(inputs, THREAD)
        if result.get("__interrupt__"):
            root.update(status="awaiting_approval")
            return s.result("awaiting_approval")
    return s.result("done")
