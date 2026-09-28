"""The shared fact-check loop as a LangGraph subgraph: judge ⇄ rewrite until supported.

The same step functions drive paper2flow's fact-check (a ``while`` loop in the plain
engine); here they are wired as a reusable subgraph so the ask agent can verify its own
answers, with every round visible in the graph's stream.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from labmate.ask.schemas import Sentence
from labmate.core.factcheck import (
    FactCheckLoop,
    finish_loop,
    judge_pending,
    rewrite_pending,
    should_rewrite,
    start_loop,
)
from labmate.core.model import LLM
from labmate.core.phases import ModelSwitcher
from labmate.core.schemas import (
    Bullet,
    ClaimCard,
    FactCheckReport,
    SlideText,
    WrittenSlides,
)

GROUP = 3
"""Sentences judged per call (the fact-check judges short blocks)."""


class VerifyState(TypedDict, total=False):
    """State of the verification subgraph."""

    loop: FactCheckLoop
    by_id: dict[str, ClaimCard]
    scope: list[list[str]]
    sentences: list[Sentence]
    report: FactCheckReport


def to_blocks(sentences: list[Sentence]) -> WrittenSlides:
    """Answer sentences as fact-check blocks of up to :data:`GROUP` sentences.

    Args:
        sentences: Answer sentences.

    Returns:
        Blocks the shared loop can check.
    """
    groups = [sentences[i : i + GROUP] for i in range(0, len(sentences), GROUP)]
    return WrittenSlides(
        hook="",
        slides=[
            SlideText(
                title="Answer",
                bullets=[Bullet(text=s.text, claim_ids=s.chunk_ids) for s in group],
            )
            for group in groups
        ],
    )


def build_verifier(
    writer: LLM, judge: LLM, switcher: ModelSwitcher, max_rounds: int = 1, workers: int = 2
) -> CompiledStateGraph:
    """Compile the judge ⇄ rewrite subgraph.

    Input: ``sentences`` and ``by_id`` (chunk id → evidence card). Output: ``sentences``
    (only the supported ones, possibly rewritten) and ``report``.

    Args:
        writer: Rewrites failing sentences.
        judge: Judges each sentence against its cited evidence (a different model).
        switcher: Model phases.
        max_rounds: Rewrite rounds after the first check.
        workers: Concurrent calls.

    Returns:
        The compiled subgraph.
    """

    def start(state: VerifyState) -> VerifyState:
        blocks = to_blocks(state["sentences"])
        scope = [sorted({c for b in s.bullets for c in b.claim_ids}) for s in blocks.slides]
        return {"loop": start_loop(blocks), "scope": scope}

    def judge_node(state: VerifyState) -> VerifyState:
        switcher.use(judge.model)
        return {"loop": judge_pending(state["loop"], state["by_id"], judge, workers)}

    def route(state: VerifyState) -> str:
        return "rewrite" if should_rewrite(state["loop"], max_rounds) else "finish"

    def rewrite_node(state: VerifyState) -> VerifyState:
        switcher.use(writer.model)
        loop = rewrite_pending(state["loop"], state["scope"], state["by_id"], writer, workers)
        return {"loop": loop}

    def finish(state: VerifyState) -> VerifyState:
        checked = finish_loop(state["loop"])
        kept = [
            Sentence(text=b.text, chunk_ids=b.claim_ids)
            for s in checked.slides.slides
            for b in s.bullets
        ]
        return {"sentences": kept, "report": checked.report}

    g = StateGraph(VerifyState)
    g.add_node("start", start)
    g.add_node("judge", judge_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("finish", finish)
    g.add_edge(START, "start")
    g.add_edge("start", "judge")
    g.add_conditional_edges("judge", route, ["rewrite", "finish"])
    g.add_edge("rewrite", "judge")
    g.add_edge("finish", END)
    return g.compile()
