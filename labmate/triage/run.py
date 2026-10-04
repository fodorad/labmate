"""triage: search arXiv, let the decider model sort the hits, and act on the choice."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import httpx
from langchain_core.runnables import RunnableConfig

from labmate.config import Config
from labmate.core.ingest import USER_AGENT, ArxivMetadata, search_arxiv
from labmate.core.lc import batch_map
from labmate.paper2flow.chain import paper2flow
from labmate.paper2post.chain import paper2post
from labmate.triage.decide import decide

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    """The decision about one paper.

    Attributes:
        id: arXiv id.
        title: Paper title.
        action: ``deep``, ``post`` or ``skip``.
        relevance: 1 to 5.
        reason: Why, in one sentence.
        output: The PDF made for it (only after a ``run``).
    """

    id: str
    title: str
    action: str
    relevance: int
    reason: str
    output: str = ""


def apply_budget(verdicts: list[Verdict], budget: int) -> list[Verdict]:
    """Keep the ``budget`` most relevant ``deep`` verdicts; the rest become ``skip``.

    The model decides what matters, code decides what is affordable. Ties keep the search order.

    Args:
        verdicts: One verdict per paper, in search order.
        budget: Most deep reads.

    Returns:
        The verdicts in the same order.
    """
    deep = sorted((v for v in verdicts if v.action == "deep"), key=lambda v: -v.relevance)
    dropped = {v.id for v in deep[budget:]}
    return [
        replace(v, action="skip", reason=f"{v.reason} (over the deep-read budget)")
        if v.id in dropped
        else v
        for v in verdicts
    ]


def write_report(verdicts: list[Verdict], out_dir: Path, topic: str) -> Path:
    """Write ``triage.md`` (a table) and ``decisions.jsonl``.

    Args:
        verdicts: The final verdicts.
        out_dir: Output folder (created).
        topic: The search topic, for the heading.

    Returns:
        The path of ``triage.md``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decisions.jsonl").write_text(
        "".join(json.dumps(asdict(v)) + "\n" for v in verdicts)
    )
    rows = [
        f"| {v.id} | {v.title} | {v.action} | {v.relevance} | {v.reason} |"
        + (f" {v.output} |" if v.output else " |")
        for v in verdicts
    ]
    table = [
        f"# Triage: {topic}",
        "",
        "| paper | title | action | relevance | reason | output |",
        "|---|---|---|---|---|---|",
        *rows,
    ]
    report = out_dir / "triage.md"
    report.write_text("\n".join(table) + "\n")
    return report


def run_triage(
    config: Config,
    topic: str,
    interests: str,
    out_dir: Path,
    limit: int = 8,
    budget: int = 2,
    run: bool = False,
    http: httpx.BaseTransport | None = None,
    ollama: httpx.BaseTransport | None = None,
) -> list[Verdict]:
    """Triage the arXiv hits for a topic.

    Args:
        config: Loaded configuration.
        topic: What to search for.
        interests: What the reader works on; the decider rates each paper against it.
        out_dir: Where ``triage.md`` and ``decisions.jsonl`` go.
        limit: Papers to look at.
        budget: Most papers that may get a deep read.
        run: Also make the overview (``deep``) or post (``post``) PDF of each chosen paper.
        http: Web transport (the network if omitted; tests).
        ollama: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        One verdict per paper, in search order.
    """
    with httpx.Client(
        transport=http or httpx.HTTPTransport(retries=2),
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as web:
        hits = search_arxiv(topic, web, limit)

    def judge(hit: tuple[str, ArxivMetadata], config_: RunnableConfig) -> Verdict:
        arxiv_id, meta = hit
        d = decide(config, meta, interests, run_config=config_, transport=ollama)
        return Verdict(arxiv_id, meta.title, d.action, d.relevance, d.reason)

    verdicts = apply_budget(
        batch_map(judge, hits, {"run_name": "triage"}, config.pipeline.workers), budget
    )
    if run:
        verdicts = [_make_output(config, v, http, ollama) for v in verdicts]
    report = write_report(verdicts, out_dir, topic)
    log.info("triage: %s", report)
    return verdicts


def _make_output(
    config: Config,
    verdict: Verdict,
    http: httpx.BaseTransport | None,
    ollama: httpx.BaseTransport | None,
) -> Verdict:
    """Run the chain that the verdict calls for (nothing for ``skip``)."""
    if verdict.action == "skip":
        return verdict
    chain = paper2flow if verdict.action == "deep" else paper2post
    pdf = chain(config, verdict.id, web=http, ollama=ollama)
    return replace(verdict, output=str(pdf))
