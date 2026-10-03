"""cv2job as a chain with one agent step: a CV and a posting in, three PDFs out.

::

    cv2job = requirements | match | gaps | tailor | letter | render

``requirements``, ``match``, ``tailor`` and ``letter`` are fixed model calls whose output code
checks against the CV and the posting. ``gaps`` is the agent: it decides how to look for
evidence the match missed and when to ask the candidate. ``render`` is plain code.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import httpx
from langchain_core.runnables import Runnable, RunnableConfig

from labmate.config import Config
from labmate.core.chat import chat_model
from labmate.core.lc import step
from labmate.cv2job.gaps import Ask, find_gaps
from labmate.cv2job.render import render_all
from labmate.cv2job.schemas import Application
from labmate.cv2job.steps import (
    load_cv,
    match_requirements,
    read_job,
    tailor_roles,
    write_highlights,
)
from labmate.paper2flow.chain import need

log = logging.getLogger(__name__)


def build_cv2job(
    config: Config, ask: Ask, out_dir: Path, today: date, ollama: httpx.BaseTransport | None = None
) -> Runnable[Application, list[Path]]:
    """The chain.

    Args:
        config: Loaded configuration.
        ask: Asks the candidate a question (for the gap agent).
        out_dir: Where the PDFs go.
        today: The current date (for the letter's years of experience).
        ollama: Transport to the Ollama server (the configured host if omitted; tests).

    Returns:
        ``Application(cv, job_text) -> [cv.pdf, cover_letter.pdf, gap_report.pdf]``.
    """
    writer = chat_model(config, config.models.text, transport=ollama)
    workers = config.pipeline.workers

    def requirements(a: Application, cfg: RunnableConfig) -> Application:
        return a.model_copy(update={"job": read_job(a.job_text, writer, cfg)})

    def match(a: Application, cfg: RunnableConfig) -> Application:
        job = need(a.job, "job")
        return a.model_copy(update={"matches": match_requirements(a.cv, job, writer, workers, cfg)})

    def gaps(a: Application, cfg: RunnableConfig) -> Application:
        job = need(a.job, "job")
        findings = find_gaps(a.cv, job, a.matches, writer, ask, cfg)
        return a.model_copy(update={"findings": findings})

    def tailor(a: Application, cfg: RunnableConfig) -> Application:
        job = need(a.job, "job")
        return a.model_copy(update={"tailored": tailor_roles(a.cv, job, writer, workers, cfg)})

    def letter(a: Application, cfg: RunnableConfig) -> Application:
        job = need(a.job, "job")
        highlights = write_highlights(a.cv, job, a.matches, a.findings, writer, cfg)
        return a.model_copy(update={"highlights": highlights})

    def render(a: Application, cfg: RunnableConfig) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "application.json").write_text(a.model_dump_json(indent=2) + "\n")
        return render_all(a, out_dir, today)

    return (
        step("requirements", requirements)
        | step("match", match)
        | step("gaps", gaps)
        | step("tailor", tailor)
        | step("letter", letter)
        | step("render", render)
    )


def cv2job(
    config: Config,
    cv_path: Path,
    job_path: Path,
    out_dir: Path,
    ask: Ask,
    today: date | None = None,
    ollama: httpx.BaseTransport | None = None,
) -> list[Path]:
    """Tailor a CV to a posting.

    Args:
        config: Loaded configuration.
        cv_path: ``cv.yaml``.
        job_path: The posting as plain text.
        out_dir: Output directory.
        ask: Asks the candidate a question (for the gap agent).
        today: The current date (default: today).
        ollama: Transport to the Ollama server (the configured host if omitted; tests).

    Returns:
        ``cv.pdf``, ``cover_letter.pdf`` and ``gap_report.pdf``.
    """
    app = Application(cv=load_cv(cv_path), job_text=job_path.read_text())
    chain = build_cv2job(config, ask, out_dir, today or date.today(), ollama)
    pdfs = chain.invoke(app, {"run_name": "cv2job"})
    log.info("cv2job: %s", ", ".join(str(p) for p in pdfs))
    return pdfs
