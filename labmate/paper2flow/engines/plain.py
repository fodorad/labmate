"""Plain-Python engine: runs the stages in order, with file checkpoints for resume.

Pipeline::

    ingest ─▶ publication ─▶ route ─▶ extract ─▶ outline ─▶ ✋ gate ─▶ write
           ─▶ fact-check ─▶ post ─▶ flows (overview ─▶ details) ─▶ render

Models are used in phases (writer, then judge, then writer) so that only one large model
is resident at a time.

Each step writes a JSON artifact into the run directory. Re-running skips steps whose
artifact exists (resume). ``fresh=True`` recomputes the LLM steps, which in ``replay``
mode proves the run reproduces from cassettes alone. The human's approved outline is an
input, not a model output, so ``fresh`` never discards it.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from labmate.config import Config, ReplayMode
from labmate.core.llm.client import OllamaClient
from labmate.paper2flow.engines.common import (
    NothingSupportedError,
    RunResult,
    Session,
    log,
    open_session,
    run_id_for,
    stage_extract,
    stage_factcheck,
    stage_flows,
    stage_ingest,
    stage_outline,
    stage_post,
    stage_publication,
    stage_render,
    stage_route,
    stage_write,
)
from labmate.paper2flow.schemas import Claims, Outline, Route
from labmate.paper2flow.steps.gate import read_gate, write_gate
from labmate.paper2flow.steps.outline import TEMPLATES

__all__ = ["NothingSupportedError", "RunResult", "run", "run_id_for"]


def gate(
    s: Session, draft: Outline, claims: Claims, route: Route, approve: bool, auto_approve: bool
) -> Outline | None:
    """The human gate: return the approved outline, or ``None`` after writing it for review.

    An approved outline (``03_outline.json``) is reused as is: it records a human decision.

    Args:
        s: Session.
        draft: The planner's outline.
        claims: Claims (to validate edits).
        route: Route (template rules for edits).
        approve: Accept ``outline.yaml`` as edited by the human.
        auto_approve: Accept the draft without review.

    Returns:
        The approved outline, or ``None`` if the run pauses for review.
    """
    approved = s.path("03_outline.json")
    gate_file = s.path("outline.yaml")
    with s.tracer.span("step.gate", approve=approve, auto_approve=auto_approve) as span:
        if not approved.exists():
            if auto_approve:
                outline = draft
            elif approve:
                outline = read_gate(gate_file, claims, route.paper_type)
            else:
                write_gate(gate_file, draft, claims, s.run_dir.name, TEMPLATES[route.paper_type])
                span.update(status="awaiting_approval")
                log.info("paused for review: %s", gate_file)
                return None
            approved.write_text(outline.model_dump_json(indent=2) + "\n")
        outline = Outline.model_validate_json(approved.read_text())
        span.update(status="approved", edited=outline != draft)
    return outline


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
    """Run the pipeline for one paper.

    Args:
        config: Loaded configuration.
        ref: arXiv id, reference or URL.
        pdf: Local PDF instead of arXiv.
        title: Title override for local PDFs.
        url: Link to a local PDF's source (shown in the outputs).
        mode: Replay mode override (defaults to ``config.replay.mode``).
        fresh: Recompute LLM steps even if their artifacts exist.
        approve: Accept the (possibly edited) ``outline.yaml`` and continue.
        auto_approve: Skip the human gate (batch runs, evaluation).
        client: Ollama client (built from config if omitted; unused in ``replay`` mode).
        http: HTTP client for arXiv (built if omitted).

    Returns:
        Status and artifact paths. ``awaiting_approval`` means ``outline.yaml`` is ready
        for review.
    """
    s = open_session(config, ref, pdf, mode, reuse=not fresh, client=client)
    root_attrs = {"paper": ref or str(pdf), "mode": s.mode.value}
    with s.tracer.span("run", engine="plain", **root_attrs) as root:
        paper = stage_publication(s, stage_ingest(s, ref, pdf, title, http, url))
        route = stage_route(s, paper)
        claims = stage_extract(s, paper)
        draft = stage_outline(s, paper, route, claims)
        outline = gate(s, draft, claims, route, approve, auto_approve)
        if outline is None:
            root.update(status="awaiting_approval")
            return s.result("awaiting_approval")
        written = stage_write(s, outline, claims)
        checked = stage_factcheck(s, written, outline, claims, paper)
        post = stage_post(s, checked, claims, paper)
        flows = stage_flows(s, outline, checked, claims, paper, route)
        return stage_render(s, paper, route, outline, checked, post, flows)
