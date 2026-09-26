"""The public gallery: publish finished runs, verify them by replay, build the static site.

A published entry (``gallery/<paper_id>/``) holds the run's step artifacts, the carousel,
the post, the trace of the run that produced it and **the cassettes of every model call
in that trace**. That makes each entry checkable by anyone: :func:`verify` re-runs the
pipeline from the cassettes alone (no model, no GPU) and compares the artifacts byte for
byte. The paper itself is not republished; it is fetched from arXiv again.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pymupdf
from pydantic import BaseModel

from paper2carousel.config import Config, ReplayMode
from paper2carousel.engines.plain import run
from paper2carousel.evals.agreement import JudgeAgreement
from paper2carousel.evals.metrics import RunMetrics, latest_completed, run_metrics
from paper2carousel.llm.replay import CassetteStore
from paper2carousel.llm.types import _cache_key
from paper2carousel.schemas import Claims, FactChecked, Paper, Post, Route, Visuals
from paper2carousel.steps.ingest import USER_AGENT, download_url, is_url
from paper2carousel.steps.render import Theme
from paper2carousel.traceview import environment, render_trace
from paper2carousel.tracing import read_trace

ARTIFACTS = [
    "01_route.json",
    "02_claims.json",
    "03_outline.draft.json",
    "03_outline.json",
    "04_slides.json",
    "05_factcheck.json",
    "06_visuals.json",
    "07_review.json",
    "08_post.json",
]
"""Step artifacts that are published and compared by :func:`verify`."""

OUTPUTS = ["carousel.pdf", "summary.pdf", "cover.png", "post.md", "summary.md", "alt_texts.json"]
"""Deliverables copied as they are (not compared: PDFs carry timestamps)."""

PAGE_DPI = 110
"""Resolution of the slide images on the site (1080 px wide pages)."""


class PublishError(RuntimeError):
    """Raised when a run cannot be published (unfinished, or cassettes missing)."""


class Meta(BaseModel):
    """What the gallery needs to know about a published paper.

    Attributes:
        paper_id: arXiv id or slug.
        title: Paper title.
        authors: Author names.
        url: Link to the paper.
        abstract: Abstract (arXiv metadata is CC0).
        source: ``arxiv`` (re-fetched when verifying) or ``pdf`` (``<paper_id>.pdf`` is
            published alongside, or downloaded again from ``url``).
        published: Date the entry was published (UTC, ISO).
    """

    paper_id: str
    title: str
    authors: list[str] = []
    url: str = ""
    abstract: str = ""
    source: str = "arxiv"
    published: str = ""


def _cassette_index(store: CassetteStore) -> dict[str, Path]:
    """Map digest-less keys to cassette files (for traces written before keys had digests)."""
    index = {}
    for path in store.root.glob("*/*.json"):
        request = dict(json.loads(path.read_text())["request"])
        request.pop("keep_alive", None)
        kind = "chat" if "messages" in request else "image"
        index[_cache_key(kind, request, None)] = path
    return index


def publish(
    run_dir: Path,
    gallery: Path,
    config: Config,
    include_pdf: bool = False,
    http: httpx.Client | None = None,
) -> tuple[Path, int]:
    """Copy a finished run, its trace and its cassettes into the gallery, then prove it replays.

    A real run is often several invocations (pause at the gate, approve, re-runs after an
    edit or an interruption), so the cassettes of every model call in the run's trace are
    candidates. The entry is then replayed with :func:`verify`; only the cassettes that
    replay actually used are kept, and an entry that doesn't reproduce is refused.

    Args:
        run_dir: Finished run (``runs/<paper_id>``).
        gallery: Gallery root.
        config: Configuration (cassette directory and lock file).
        include_pdf: Also publish the paper PDF (only for papers you may redistribute).
        http: HTTP client for re-fetching the paper during the check (tests).

    Returns:
        The entry directory and the number of cassettes kept.

    Raises:
        PublishError: If the run hasn't finished or the entry does not replay exactly.
    """
    if not (run_dir / "carousel.pdf").exists() or not (run_dir / "05_factcheck.json").exists():
        raise PublishError(f"{run_dir} has not finished (run it with --approve first)")
    if not latest_completed(run_dir / "trace.jsonl"):
        raise PublishError(f"{run_dir} has no completed run in trace.jsonl")
    spans = read_trace(run_dir / "trace.jsonl")
    store = CassetteStore(config.replay.dir)
    keys = sorted({str(s["key"]) for s in spans if str(s["name"]).startswith("llm.")})
    found: dict[str, Path] = {k: store.path(k) for k in keys if store.path(k).exists()}
    if len(found) < len(keys):
        legacy = _cassette_index(store)
        found |= {k: legacy[k] for k in keys if k not in found and k in legacy}

    paper = Paper.model_validate_json((run_dir / "00_paper.json").read_text())
    entry = gallery / run_dir.name
    if entry.exists():
        shutil.rmtree(entry)
    entry.mkdir(parents=True)
    for name in [*ARTIFACTS, *OUTPUTS]:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, entry / name)
    for path in found.values():
        dest = entry / "cassettes" / path.parent.name / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    if config.replay.lock_file.exists():
        shutil.copy2(config.replay.lock_file, entry / "models.lock")
    for span in spans:  # the root span holds the input path, which may be a local path
        if span["name"] == "run":
            span["paper"] = paper.paper_id
    (entry / "trace.jsonl").write_text("".join(json.dumps(s) + "\n" for s in spans))
    source = "arxiv" if paper.url.startswith("https://arxiv.org/") else "pdf"
    if include_pdf:
        shutil.copy2(run_dir / "paper.pdf", entry / f"{paper.paper_id}.pdf")
    meta = Meta(
        paper_id=paper.paper_id,
        title=paper.title,
        authors=paper.authors,
        url=paper.url,
        abstract=paper.abstract,
        source=source,
        published=datetime.now(UTC).date().isoformat(),
    )
    (entry / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n")

    report = verify(entry, config, http)
    if not report.ok:
        problem = report.error or f"different: {', '.join(report.different)}"
        raise PublishError(f"{entry} does not replay from its cassettes ({problem})")
    for path in (entry / "cassettes").glob("*/*.json"):
        if path.stem not in report.used_keys:
            path.unlink()
    for shard in (entry / "cassettes").glob("*"):
        if shard.is_dir() and not any(shard.iterdir()):
            shard.rmdir()
    return entry, len(report.used_keys)


class VerifyReport(BaseModel):
    """Outcome of replaying one gallery entry.

    Attributes:
        paper_id: Entry id.
        identical: Artifacts reproduced byte for byte.
        different: Artifacts that differ from the published ones.
        error: Why the replay stopped, if it did (e.g. a missing cassette).
        used_keys: Cassettes the replay read.
    """

    paper_id: str
    identical: list[str] = []
    different: list[str] = []
    error: str = ""
    used_keys: list[str] = []

    @property
    def ok(self) -> bool:
        """True if the replay finished and every artifact matched."""
        return not self.error and not self.different


def verify(entry: Path, config: Config, http: httpx.Client | None = None) -> VerifyReport:
    """Re-run a published entry from its cassettes only and compare the artifacts.

    The approved outline is taken from the entry (it records the human gate decision);
    everything else is recomputed.

    Args:
        entry: Gallery entry directory.
        config: Base configuration (models, generation settings).
        http: HTTP client for arXiv (tests).

    Returns:
        Which artifacts matched.
    """
    meta = Meta.model_validate_json((entry / "meta.json").read_text())
    report = VerifyReport(paper_id=meta.paper_id)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = config.model_copy(deep=True)
        cfg.replay.dir = entry / "cassettes"
        cfg.replay.lock_file = entry / "models.lock"
        cfg.tracing.runs_dir = Path(tmp) / "runs"
        run_dir = cfg.tracing.runs_dir / meta.paper_id
        run_dir.mkdir(parents=True)
        shutil.copy2(entry / "03_outline.json", run_dir / "03_outline.json")
        pdf = None
        try:
            if meta.source == "pdf":
                pdf = Path(tmp) / f"{meta.paper_id}.pdf"
                published = entry / f"{meta.paper_id}.pdf"
                if published.exists():
                    shutil.copy2(published, pdf)
                elif is_url(meta.url):
                    own = http is None
                    client = http or httpx.Client(
                        timeout=120, follow_redirects=True, headers={"User-Agent": USER_AGENT}
                    )
                    try:
                        pdf.write_bytes(
                            download_url(meta.url, Path(tmp) / "dl", client).read_bytes()
                        )
                    finally:
                        if own:
                            client.close()
                else:
                    raise FileNotFoundError(f"no PDF published and no URL to fetch {meta.paper_id}")
            run(
                cfg,
                ref=meta.paper_id if pdf is None else None,
                pdf=pdf,
                title=meta.title,
                url=meta.url,
                mode=ReplayMode.REPLAY,
                fresh=True,
                approve=True,
                http=http,
            )
        except Exception as e:  # noqa: BLE001 - reported, not raised
            report.error = f"{type(e).__name__}: {e}"
        replay_trace = run_dir / "trace.jsonl"
        if replay_trace.exists():
            report.used_keys = sorted(
                {str(sp["key"]) for sp in read_trace(replay_trace) if "key" in sp}
            )
        for name in ARTIFACTS:
            if not (entry / name).exists():
                continue
            same = (run_dir / name).exists() and (run_dir / name).read_bytes() == (
                entry / name
            ).read_bytes()
            (report.identical if same else report.different).append(name)
    return report


# --- static site ---------------------------------------------------------------------------


class SlideEvidence(BaseModel):
    """A published slide with the evidence behind each bullet."""

    title: str
    bullets: list[dict[str, Any]]


def _entries(gallery: Path) -> list[Path]:
    return sorted(p.parent for p in gallery.glob("*/meta.json"))


def _page_images(pdf: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            name = f"page-{i:02d}.png"
            page.get_pixmap(dpi=PAGE_DPI).save(out_dir / name)
            names.append(name)
    return names


def _load(path: Path, model: Callable[[str], Any]) -> Any:
    return model(path.read_text()) if path.exists() else None


def paper_context(entry: Path) -> dict[str, Any]:
    """Everything the paper page shows, from the published artifacts.

    Args:
        entry: Gallery entry directory.

    Returns:
        Template context: meta, metrics, route, slides with evidence, the fact-check
        audit, the visuals agent's tool calls, the post and alt texts.
    """
    meta = Meta.model_validate_json((entry / "meta.json").read_text())
    claims = Claims.model_validate_json((entry / "02_claims.json").read_text())
    checked = FactChecked.model_validate_json((entry / "05_factcheck.json").read_text())
    by_id = {c.id: c for c in claims.cards}
    slides = [
        SlideEvidence(
            title=s.title,
            bullets=[
                {"text": b.text, "evidence": [by_id[i] for i in b.claim_ids if i in by_id]}
                for b in s.bullets
            ],
        )
        for s in checked.slides.slides
    ]
    report = checked.report
    last = {(c.slide, c.bullet): c for c in report.rounds[-1]} if report.rounds else {}
    audit = []
    for c in report.rounds[0] if report.rounds else []:
        if c.passed:
            continue
        final = last.get((c.slide, c.bullet))
        if c.slide in report.dropped_slides:
            outcome = "slide dropped"
        elif final is not None and not final.passed:
            outcome = "dropped"
        else:
            outcome = f"rewritten: {final.text}" if final and final.text != c.text else "passed"
        audit.append({"check": c, "outcome": outcome})
    alt = (
        json.loads((entry / "alt_texts.json").read_text())
        if (entry / "alt_texts.json").exists()
        else []
    )
    return {
        "meta": meta,
        "metrics": run_metrics(entry),
        "route": _load(entry / "01_route.json", Route.model_validate_json),
        "slides": slides,
        "hook": checked.slides.hook,
        "audit": audit,
        "rounds": len(report.rounds),
        "visuals": _load(entry / "06_visuals.json", Visuals.model_validate_json),
        "post": _load(entry / "08_post.json", Post.model_validate_json),
        "post_text": (entry / "post.md").read_text() if (entry / "post.md").exists() else "",
        "alt": {a["page"]: a["alt_text"] for a in alt},
    }


def build_site(
    gallery: Path,
    out: Path,
    judges: Path | None = None,
    repo_url: str = "https://github.com/fodorad/paper2carousel",
) -> list[RunMetrics]:
    """Generate the static gallery site.

    Args:
        gallery: Gallery root with published entries.
        out: Output directory (deployed to GitHub Pages).
        judges: Optional ``judges.json`` from ``make judges`` to show judge agreement.
        repo_url: Link to the repository.

    Returns:
        Metrics of the entries, in site order.
    """
    env = environment()
    theme = Theme()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / ".nojekyll").write_text("")
    papers = []
    for entry in _entries(gallery):
        ctx = paper_context(entry)
        dest = out / entry.name
        pages = _page_images(entry / "carousel.pdf", dest / "pages")
        for name in ["carousel.pdf", "summary.pdf", "post.md", "summary.md"]:
            if (entry / name).exists():
                shutil.copy2(entry / name, dest / name)
        spans = [json.loads(line) for line in (entry / "trace.jsonl").read_text().splitlines()]
        (dest / "trace.html").write_text(render_trace(spans, f"Trace · {ctx['meta'].title}", theme))
        page = env.get_template("paper.html.j2").render(
            theme=theme,
            pages=pages,
            repo_url=repo_url,
            has_summary_pdf=(entry / "summary.pdf").exists(),
            **ctx,
        )
        (dest / "index.html").write_text(page)
        papers.append(
            {
                "meta": ctx["meta"],
                "metrics": ctx["metrics"],
                "thumb": f"{entry.name}/pages/{pages[0]}",
                "route": ctx["route"],
            }
        )
    agreement = (
        [JudgeAgreement.model_validate(j) for j in json.loads(judges.read_text())]
        if judges is not None and judges.exists()
        else []
    )
    index = env.get_template("index.html.j2").render(
        theme=theme, papers=papers, agreement=agreement, repo_url=repo_url
    )
    (out / "index.html").write_text(index)
    return [p["metrics"] for p in papers]
