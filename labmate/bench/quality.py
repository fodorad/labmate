"""Quality scores for a run, from checks the code already has. No extra model is asked.

Each score is the mean of a few parts between 0 and 1, shown as a percentage; the report lists the
parts, so the trade-off between speed and quality can be read off. The scout score is a proxy
(there is no answer key for a research topic).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from labmate.ask.evals import AnswerCase
from labmate.cv2job.schemas import Application
from labmate.cv2job.steps import covered_requirements
from labmate.paper2flow.evals.metrics import RunMetrics
from labmate.paper2post.schemas import Post

CV_KEY = {"PyTorch": True, "Python": True, "Docker": True, "papers": True, "Kubernetes": False}
"""The example posting's answer key: does the example CV cover the requirement naming this term?"""


@dataclass(frozen=True)
class Quality:
    """A score and what it is made of.

    Attributes:
        score: The mean of the parts, in percent.
        parts: Part name to a value between 0 and 1.
    """

    score: float
    parts: dict[str, float] = field(default_factory=dict)


def _mean(parts: dict[str, float]) -> Quality:
    return Quality(round(100 * sum(parts.values()) / len(parts), 1), parts)


def ask_quality(cases: list[AnswerCase]) -> Quality:
    """Score ask answers.

    Args:
        cases: Scored golden questions (see :func:`labmate.ask.evals.score_answer`).

    Returns:
        Parts: right answer or refusal, expected source cited, share of drafted sentences kept.
    """
    hits = [c.source_hit for c in cases if c.source_hit is not None]
    kept, dropped = sum(c.sentences for c in cases), sum(c.dropped for c in cases)
    return _mean(
        {
            "right answer or refusal": sum(c.abstain_ok for c in cases) / len(cases),
            "expected source cited": sum(hits) / len(hits) if hits else 1.0,
            "sentences kept": kept / (kept + dropped) if kept + dropped else 1.0,
        }
    )


def cv2job_quality(app: Application, produced: int) -> Quality:
    """Score a cv2job run against the example's answer key.

    Args:
        app: The finished application.
        produced: How many of the three PDFs were written.

    Returns:
        Parts: requirements classified right (covered or gap) and PDFs produced.
    """
    if app.job is None:
        return _mean({"requirements classified right": 0.0, "PDFs produced": produced / 3})
    covered = {r.id for r, _, _ in covered_requirements(app.job, app.matches, app.findings)}
    right = 0
    for term, should_cover in CV_KEY.items():
        found = [r for r in app.job.requirements if term.lower() in r.text.lower()]
        right += bool(found) and (found[0].id in covered) == should_cover
    return _mean(
        {"requirements classified right": right / len(CV_KEY), "PDFs produced": produced / 3}
    )


def scout_quality(notes: str | None, cited: int) -> Quality:
    """Score a scout run (a proxy: it finished, and how many papers it read and cited).

    Args:
        notes: The notes, or ``None`` if the agent stopped without writing them.
        cited: Papers cited (each was read, by the citation guard).

    Returns:
        Parts: notes written, papers cited (up to five count).
    """
    return _mean(
        {"notes written": float(notes is not None), "papers cited (of 5)": min(cited, 5) / 5}
    )


def flow_quality(metrics: RunMetrics) -> Quality:
    """Score a paper2flow run from its fact-check record.

    Args:
        metrics: The run's metrics.

    Returns:
        Parts: first-draft bullets that passed, claims kept, cards that survived.
    """
    first = metrics.bullets_first
    claims = metrics.claims_verified + metrics.claims_rejected
    return _mean(
        {
            "first-draft bullets passing": 1 - metrics.unsupported_first / first if first else 0.0,
            "claims kept": metrics.claims_verified / claims if claims else 0.0,
            "cards kept": metrics.cards / 4,
        }
    )


def post_quality(post: Post) -> Quality:
    """Score a paper2post run from its fact-check record.

    Args:
        post: The finished post.

    Returns:
        Parts: first-draft sentences that passed and sentences kept.
    """
    drafted = post.report.total_first
    return _mean(
        {
            "first-draft sentences passing": 1 - post.report.failed_first / drafted
            if drafted
            else 0.0,
            "sentences kept": len(post.takeaways) / drafted if drafted else 0.0,
        }
    )
