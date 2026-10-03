"""Evaluation of ask: the graph and the agent on a golden set you write yourself.

``library/golden.yaml`` lists questions, the sources a good answer cites, and which
questions the library cannot answer (the answer should be a refusal). Each question runs
through both :mod:`labmate.ask.graph` and :mod:`labmate.ask.agent`; the report scores
refusals, cited sources and how many drafted sentences the evidence checks removed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from labmate.ask.schemas import Answer


class GoldenQuestion(BaseModel):
    """One hand-written test question.

    Attributes:
        question: The question.
        sources: Source ids a good answer cites (any of them); empty = don't check.
        abstain: True if the library does not answer it and the agent should say so.
    """

    question: str
    sources: list[str] = Field(default_factory=list)
    abstain: bool = False


class AnswerCase(BaseModel):
    """How one agent did on one golden question.

    Attributes:
        question: The question.
        agent: ``graph`` or ``agent``.
        abstained: Whether it declined to answer.
        abstain_ok: Declined exactly when it should have.
        source_hit: Cited an expected source (``None`` if not applicable).
        sentences: Cited sentences in the answer.
        dropped: Drafted sentences the evidence checks removed.
        seconds: Wall time.
        answer: The answer text.
    """

    question: str
    agent: str
    abstained: bool
    abstain_ok: bool
    source_hit: bool | None
    sentences: int
    dropped: int
    seconds: float
    answer: str


def load_golden(path: Path) -> list[GoldenQuestion]:
    """Read the golden set (a YAML list of questions).

    Args:
        path: ``library/golden.yaml``.

    Returns:
        The questions.
    """
    import yaml  # noqa: PLC0415 - only the evaluation needs it

    return [GoldenQuestion.model_validate(q) for q in yaml.safe_load(path.read_text()) or []]


def score_answer(golden: GoldenQuestion, answer: Answer, started: float) -> AnswerCase:
    """Score one answer.

    Args:
        golden: The question and what a good answer looks like.
        answer: The agent's answer.
        started: ``time.perf_counter()`` when the question was asked.

    Returns:
        The case.
    """
    cited = {c.source_id for c in answer.citations}
    return AnswerCase(
        question=golden.question,
        agent=answer.agent,
        abstained=answer.abstained,
        abstain_ok=answer.abstained == golden.abstain,
        source_hit=bool(cited & set(golden.sources))
        if golden.sources and not golden.abstain
        else None,
        sentences=len(answer.sentences),
        dropped=answer.dropped,
        seconds=round(time.perf_counter() - started, 1),
        answer=answer.text,
    )


def answer_markdown(cases: Sequence[AnswerCase]) -> str:
    """Results table per agent.

    Args:
        cases: Scored answers of both agents.

    Returns:
        Markdown.
    """
    rows = [
        "| Agent | Correct answer/refusal | Cited an expected source | Cited sentences "
        "| Sentences dropped | Time (mean) |",
        "|---|---|---|---|---|---|",
    ]
    for agent in dict.fromkeys(c.agent for c in cases):
        mine = [c for c in cases if c.agent == agent]
        hits = [c.source_hit for c in mine if c.source_hit is not None]
        rows.append(
            f"| {agent} | {sum(c.abstain_ok for c in mine)}/{len(mine)} | "
            f"{sum(hits)}/{len(hits)} | {sum(c.sentences for c in mine)} | "
            f"{sum(c.dropped for c in mine)} | "
            f"{sum(c.seconds for c in mine) / len(mine):.0f} s |"
        )
    return "\n".join(rows) + "\n"
