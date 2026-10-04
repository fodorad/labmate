"""Run each use case on a small fixed workload, on one model profile, and measure it.

Every cell starts cold (all models unloaded) with the reply cache off, so it pays what a first
request pays. The workloads are small on purpose: four golden questions for ask, the example CV for
cv2job, one topic for scout, and one paper for paper2flow and paper2post (only with ``--full``).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import httpx

from labmate.ask.agent import ask_agent, build_agent
from labmate.ask.evals import load_golden, score_answer
from labmate.ask.graph import ask, compile_graph
from labmate.ask.session import open_ask
from labmate.bench.cells import Cell
from labmate.bench.micro import unload_all
from labmate.bench.quality import (
    Quality,
    ask_quality,
    cv2job_quality,
    flow_quality,
    post_quality,
    scout_quality,
)
from labmate.config import CacheConfig, Config, apply_profile
from labmate.core.usage import UsageCollector, ollama_resident
from labmate.cv2job.chain import cv2job
from labmate.cv2job.schemas import Application
from labmate.paper2flow.chain import paper2flow
from labmate.paper2flow.evals.metrics import run_metrics
from labmate.paper2post.chain import POST_ARTIFACT, paper2post
from labmate.paper2post.schemas import Post
from labmate.scout.agent import cited_ids, run_scout

ASK_QUESTIONS = (0, 2, 5, 11)
"""Golden questions used: a definition, a number, a list of datasets, and an off-topic one."""

SCOUT_TOPIC = "linear attention for long sequences"
PAPER = "1706.03762"
EXAMPLES = Path("examples/cv2job")

USE_CASES = ("ask: graph", "ask: agent", "cv2job", "scout")
FULL_USE_CASES = (*USE_CASES, "paper2flow", "paper2post")


def run_cell(use_case: str, profile: str, base: Config) -> Cell:
    """Run one use case on one profile, cold, and measure it.

    Args:
        use_case: One of :data:`FULL_USE_CASES`.
        profile: A key of ``[profiles]``.
        base: Loaded configuration.

    Returns:
        The measured cell; ``error`` is set if the run failed.
    """
    config = base.model_copy(deep=True)
    apply_profile(config, profile)
    collector = UsageCollector(ollama_resident(config.ollama.host))
    config.cache, config.callbacks = CacheConfig(enabled=False), [collector]
    cell = Cell(use_case, profile, config.models.text, config.models.critic)
    unload_all(config.ollama.host)
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            quality, divisor = _RUNNERS[use_case](config, Path(tmp))
    except (httpx.HTTPError, ValueError, RuntimeError, OSError) as e:
        cell.error = f"{type(e).__name__}: {e}"[:160]
        return cell
    cell.seconds = (time.perf_counter() - started) / divisor
    usage = collector.summary()
    cell.calls, cell.tokens_in, cell.tokens_out = usage.calls, usage.tokens_in, usage.tokens_out
    cell.load_s, cell.swaps, cell.peak_gb = usage.load_s, usage.swaps, usage.peak_gb
    cell.quality, cell.parts = quality.score, quality.parts
    return cell


def _ask_graph(config: Config, tmp: Path) -> tuple[Quality, int]:
    s = open_ask(config)
    try:
        golden = load_golden(config.ask.library / "golden.yaml")
        graph, cases = compile_graph(s), []
        for i in ASK_QUESTIONS:
            started = time.perf_counter()
            cases.append(score_answer(golden[i], ask(graph, golden[i].question), started))
    finally:
        s.close()
    return ask_quality(cases), len(ASK_QUESTIONS)


def _ask_agent(config: Config, tmp: Path) -> tuple[Quality, int]:
    s = open_ask(config)
    try:
        golden = load_golden(config.ask.library / "golden.yaml")
        agent, cases = build_agent(s), []
        for i in ASK_QUESTIONS:
            started = time.perf_counter()
            cases.append(score_answer(golden[i], ask_agent(s, agent, golden[i].question), started))
    finally:
        s.close()
    return ask_quality(cases), len(ASK_QUESTIONS)


def _cv2job(config: Config, tmp: Path) -> tuple[Quality, int]:
    pdfs = cv2job(config, EXAMPLES / "cv.yaml", EXAMPLES / "job.txt", tmp, ask=lambda q: "")
    app = Application.model_validate_json((tmp / "application.json").read_text())
    return cv2job_quality(app, sum(p.exists() for p in pdfs)), 1


def _scout(config: Config, tmp: Path) -> tuple[Quality, int]:
    notes = run_scout(config, SCOUT_TOPIC, tmp, max_deep=0)
    text = notes.read_text() if notes else None
    return scout_quality(text, len(cited_ids(text)) if text else 0), 1


def _paper2flow(config: Config, tmp: Path) -> tuple[Quality, int]:
    config.tracing.runs_dir = tmp
    pdf = paper2flow(config, PAPER)
    return flow_quality(run_metrics(pdf.parent)), 1


def _paper2post(config: Config, tmp: Path) -> tuple[Quality, int]:
    config.tracing.runs_dir = tmp
    pdf = paper2post(config, PAPER)
    post = Post.model_validate(json.loads((pdf.parent / POST_ARTIFACT).read_text()))
    return post_quality(post), 1


_RUNNERS = {
    "ask: graph": _ask_graph,
    "ask: agent": _ask_agent,
    "cv2job": _cv2job,
    "scout": _scout,
    "paper2flow": _paper2flow,
    "paper2post": _paper2post,
}
