"""The ask graph: corrective RAG as a LangGraph ``StateGraph``.

.. code-block:: text

    understand ─┬─(off topic)─────────────────────────────────────────────▶ abstain
                ├─(ambiguous)─▶ ✋ clarify ─┐
                └──────────────────────────┴─▶ retrieve ─▶ grade ─┬─(enough)─▶ answer ─▶ verify
                                                  ▲               │
                                                  └── rewrite ◀───┘ (not enough)
                                             grade ─(nothing relevant)─▶ abstain

The code decides the path; the model fills in each node: it reads the question, grades what
was retrieved, rewrites the query, writes the cited answer and, as a second model, checks it.
A graph rather than a chain because the retrieve-grade-rewrite cycle has a state-dependent
exit and the graph can stop to ask which reading of an ambiguous question is meant
(``interrupt``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from labmate.ask.answer import (
    NOT_FOUND,
    OFF_TOPIC,
    compose,
    structure_problems,
)
from labmate.ask.evidence import format_evidence
from labmate.ask.index import Hit
from labmate.ask.prompts import load_prompt
from labmate.ask.schemas import (
    Answer,
    AnswerDraft,
    Grade,
    Rewrite,
    Understanding,
)
from labmate.ask.session import AskSession
from labmate.ask.verify import verify_sentences
from labmate.core.lc import prompt, structured


class AskState(TypedDict, total=False):
    """Graph state: one question from understanding to answer."""

    question: str
    understanding: Understanding
    query: str
    tried: list[str]
    loop: int
    candidates: list[str]
    relevant: list[str]
    sufficient: bool
    missing: str
    draft: AnswerDraft
    answer: Answer


# --- steps (plain functions, testable without the graph) -----------------------------------


def check_understanding(u: Understanding) -> list[str]:
    """Rules for :class:`Understanding`.

    Args:
        u: The model's reading of the question.

    Returns:
        Problems; empty if valid.
    """
    problems = []
    if not u.search.strip():
        problems.append("search must repeat the question as a query")
    if len(u.options) == 1 or len(u.options) > 4:
        problems.append("options: give 2 to 4 readings, or none if the question is clear")
    return problems


def understand(s: AskSession, question: str) -> Understanding:
    """Classify the question and turn it into a search query.

    Args:
        s: Session.
        question: The user's question.

    Returns:
        The understanding.
    """
    sources = "\n".join(f"- {x.name}: {x.title}" for x in s.index.sources())
    chain = prompt(load_prompt("understand")) | structured(
        s.writer, Understanding, check=check_understanding
    )
    return chain.invoke({"question": question, "sources": sources or "(none)"})


def retrieve(s: AskSession, query: str, exclude: list[str]) -> list[Hit]:
    """Hybrid search, skipping chunks already judged relevant.

    Args:
        s: Session.
        query: Search query.
        exclude: Chunk ids to leave out.

    Returns:
        Up to ``top_k`` hits.
    """
    k = s.config.ask.top_k
    hits = s.index.search(query, s.embedder.query(query), k + len(exclude))
    return [h for h in hits if h.chunk.id not in exclude][:k]


def grade(s: AskSession, question: str, query: str, candidates: list[str]) -> Grade:
    """The judge model decides which candidates help and whether they suffice.

    Args:
        s: Session.
        question: The whole question (for context).
        query: The search query.
        candidates: Chunk ids to grade.

    Returns:
        The grade.
    """

    def check(g: Grade) -> list[str]:
        unknown = sorted(set(g.relevant) - set(candidates))
        return [f"relevant lists ids that were not shown: {unknown}"] if unknown else []

    chain = prompt(load_prompt("grade")) | structured(s.judge, Grade, check=check)
    return chain.invoke(
        {"question": question, "query": query, "evidence": format_evidence(s.index, candidates)}
    )


def rewrite_query(s: AskSession, query: str, missing: str, tried: list[str]) -> str:
    """A new search query for what is still missing.

    Args:
        s: Session.
        query: The query so far.
        missing: What the grader found missing.
        tried: Queries already used.

    Returns:
        The new query.
    """

    def check(r: Rewrite) -> list[str]:
        if r.query.strip().lower() in {t.strip().lower() for t in tried}:
            return ["that query was already tried; phrase it differently"]
        return ["at most 20 words"] if len(r.query.split()) > 20 else []

    chain = prompt(load_prompt("rewrite_query")) | structured(s.writer, Rewrite, check=check)
    return chain.invoke(
        {
            "query": query,
            "missing": missing or "(not stated)",
            "tried": "\n".join(f"- {q}" for q in tried),
        }
    ).query


def write_answer(s: AskSession, question: str, chunk_ids: list[str]) -> AnswerDraft:
    """The writer answers from the evidence, every sentence citing its chunks.

    Args:
        s: Session.
        question: The question.
        chunk_ids: Evidence chunks.

    Returns:
        The draft.
    """
    known = set(chunk_ids)

    def check(d: AnswerDraft) -> list[str]:
        return structure_problems(d.sentences, known)

    chain = prompt(load_prompt("answer")) | structured(s.writer, AnswerDraft, check=check)
    return chain.invoke({"question": question, "evidence": format_evidence(s.index, chunk_ids)})


# --- the graph ------------------------------------------------------------------------------


def build_graph(s: AskSession) -> StateGraph:
    """Wire the ask graph (not compiled; see :func:`compile_graph`).

    Args:
        s: Session (closed over by the nodes; not part of the state).

    Returns:
        The graph builder.
    """
    max_loops = s.config.ask.max_loops

    def understand_node(state: AskState) -> AskState:
        u = understand(s, state["question"])
        return {"understanding": u, "query": u.search, "tried": [], "loop": 0, "relevant": []}

    def route(state: AskState) -> str:
        u = state["understanding"]
        if not u.on_topic:
            return "abstain"
        return "clarify" if u.options else "retrieve"

    def clarify(state: AskState) -> AskState:
        u = state["understanding"]
        choice = interrupt({"question": state["question"], "options": u.options})
        return {"query": f"{u.search} ({choice})"}

    def retrieve_node(state: AskState) -> AskState:
        hits = retrieve(s, state["query"], state["relevant"])
        return {"candidates": [h.chunk.id for h in hits]}

    def grade_node(state: AskState) -> AskState:
        g = grade(s, state["question"], state["query"], state["candidates"])
        relevant = list(dict.fromkeys([*state["relevant"], *g.relevant]))
        return {"relevant": relevant, "sufficient": g.sufficient, "missing": g.missing,
                "loop": state["loop"] + 1}  # fmt: skip

    def decide(state: AskState) -> str:
        if not state["sufficient"] and state["loop"] < max_loops:
            return "rewrite"
        return "answer" if state["relevant"] else "abstain"

    def rewrite_node(state: AskState) -> AskState:
        tried = [*state["tried"], state["query"]]
        return {"query": rewrite_query(s, state["query"], state["missing"], tried),
                "tried": tried}  # fmt: skip

    def answer_node(state: AskState) -> AskState:
        return {"draft": write_answer(s, state["question"], state["relevant"])}

    def verify_node(state: AskState) -> AskState:
        question, sentences = state["question"], state["draft"].sentences
        kept, dropped = verify_sentences(s, question, sentences)
        return {"answer": compose(s.index, question, kept, dropped)}

    def abstain(state: AskState) -> AskState:
        off = not state["understanding"].on_topic
        answer = Answer(
            question=state["question"],
            text=OFF_TOPIC if off else NOT_FOUND,
            abstained=True,
            reason="off topic" if off else "no relevant evidence",
        )
        return {"answer": answer}

    g = StateGraph(AskState)
    for name, fn in [
        ("understand", understand_node),
        ("clarify", clarify),
        ("retrieve", retrieve_node),
        ("grade", grade_node),
        ("rewrite", rewrite_node),
        ("answer", answer_node),
        ("verify", verify_node),
        ("abstain", abstain),
    ]:
        g.add_node(name, fn)  # type: ignore[arg-type,call-overload,unused-ignore]
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", route, ["abstain", "clarify", "retrieve"])
    g.add_edge("clarify", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", decide, ["rewrite", "answer", "abstain"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", "verify")
    g.add_edge("verify", END)
    g.add_edge("abstain", END)
    return g


def compile_graph(s: AskSession) -> CompiledStateGraph:
    """Compile the graph with an in-memory checkpointer (needed to pause at ``interrupt``).

    Args:
        s: Session.

    Returns:
        The compiled graph.
    """
    return build_graph(s).compile(checkpointer=InMemorySaver())


Chooser = Callable[[str, list[str]], str]
"""Picks one reading of an ambiguous question: ``(question, options) -> option``."""


def ask(graph: CompiledStateGraph, question: str, choose: Chooser | None = None) -> Answer:
    """Ask one question.

    Args:
        graph: The compiled graph.
        question: The question.
        choose: Resolves a clarification interrupt (default: the first option).

    Returns:
        The answer.
    """
    config: RunnableConfig = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = graph.invoke({"question": question}, config)
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        options = list(payload["options"])
        choice = choose(payload["question"], options) if choose else options[0]
        result = graph.invoke(Command(resume=choice), config)
    answer: Answer = result["answer"]
    return answer
