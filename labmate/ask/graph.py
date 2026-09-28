"""The ask agent as a LangGraph ``StateGraph``.

.. code-block:: text

    understand ─┬─(off topic)────────────────────────────────────────────▶ abstain
                ├─(ambiguous)─▶ ✋ clarify ─┐
                └──────────────────────────┴─▶ plan ─Send × k─▶ research ─┐
                                                                         │
       research (subgraph, per query):  retrieve ─▶ grade ─┬─▶ done       │
                                            ▲              │ not enough   │
                                            └── rewrite ◀──┘ (widen tier) │
                                                                         ▼
          conflicts ─▶ answer ─▶ verify (fact-check subgraph) ─▶ finalize
                 └─(no evidence)──────────────────────────────▶ abstain

Why a graph and not a chain: the number of research branches is decided at run time
(``Send``), each branch loops until its evidence is good enough (a cycle with a
state-dependent exit), verification is a reusable subgraph, the graph can stop to ask
which reading of an ambiguous question is meant (``interrupt``), and a checkpointer
carries the conversation across turns.

Retrieval is tiered: the dissertation (tier 1) is searched first and is the source of
truth; the author's papers (tier 2) are added only when the dissertation is not enough;
outside sources (tier 3) only for questions about related work.
"""

from __future__ import annotations

import operator
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel

from labmate.ask import schemas
from labmate.ask.evidence import evidence_cards, format_evidence, label
from labmate.ask.index import Hit
from labmate.ask.prompts import load_prompt
from labmate.ask.schemas import (
    Answer,
    AnswerDraft,
    Citation,
    Conflict,
    Conflicts,
    Finding,
    Grade,
    Intent,
    Plan,
    Rewrite,
    Sentence,
    Turn,
    Understanding,
)
from labmate.ask.session import AskSession
from labmate.ask.verify import build_verifier
from labmate.core import schemas as core_schemas
from labmate.core.factcheck import FactCheckLoop, numbers_in
from labmate.core.llm.structured import structured_chat

START_TIERS: dict[str, list[int]] = {"thesis": [1], "own_work": [1], "related": [1, 2, 3]}
"""Tiers searched first, per intent: the dissertation first, outside sources only for
related work."""

MAX_SENTENCE_WORDS = 35
"""Longest answer sentence."""

HISTORY_TURNS = 4
"""Previous turns shown when making a follow-up question self-contained."""

OFF_TOPIC = (
    "That question is outside my library (my PhD dissertation and papers), so I can't "
    "answer it from sources."
)
NOT_FOUND = "My dissertation and papers don't answer that, as far as I can find."


def collect(old: list[Finding] | None, new: list[Finding] | None) -> list[Finding]:
    """Reducer of ``findings``: ``None`` starts a new turn, a list is appended.

    Args:
        old: Findings so far.
        new: New findings, or ``None`` to reset.

    Returns:
        The merged list.
    """
    return [] if new is None else [*(old or []), *new]


class AskState(TypedDict, total=False):
    """Graph state; ``history`` persists across turns through the checkpointer."""

    question: str
    history: Annotated[list[Turn], operator.add]
    understanding: Understanding
    queries: list[str]
    findings: Annotated[list[Finding] | None, collect]
    conflicts: list[Conflict]
    draft: AnswerDraft
    sentences: list[Sentence]
    dropped: int
    answer: Answer


class ResearchState(TypedDict, total=False):
    """State of the research subgraph (one per planned query)."""

    planned: str
    query: str
    intent: str
    tiers: list[int]
    loop: int
    tried: list[str]
    candidates: list[str]
    relevant: list[str]
    sufficient: bool
    missing: str
    finding: Finding


# --- steps (plain functions, testable without the graph) -----------------------------------


def _emit(event: str, **data: Any) -> None:
    """Send a custom stream event (the dashboard shows them; no-op outside streaming)."""
    try:
        get_stream_writer()({"event": event, **data})
    except RuntimeError:  # called outside a graph run
        pass


def check_understanding(u: Understanding) -> list[str]:
    """Rules for :class:`Understanding`.

    Args:
        u: The model's reading of the question.

    Returns:
        Problems; empty if valid.
    """
    problems = []
    if not u.standalone.strip():
        problems.append("standalone must repeat the question when nothing needs resolving")
    if len(u.options) == 1 or len(u.options) > 4:
        problems.append("options: give 2 to 4 readings, or none if the question is clear")
    return problems


def understand(s: AskSession, question: str, history: list[Turn]) -> Understanding:
    """Make the question self-contained and classify it.

    Args:
        s: Session.
        question: The user's message.
        history: Previous turns of the conversation.

    Returns:
        The understanding.
    """
    turns = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in history[-HISTORY_TURNS:])
    sources = "\n".join(f"- {x.name} (tier {x.tier}): {x.title}" for x in s.index.sources())
    prompt = load_prompt("understand").format(
        question=question, history=turns or "(none)", sources=sources or "(none)"
    )
    s.switcher.use(s.llm.model)
    return structured_chat(
        s.llm.backend, s.llm.request(prompt), Understanding, check=check_understanding
    )


def plan_queries(s: AskSession, u: Understanding) -> list[str]:
    """Split the question into 1–4 search queries.

    Args:
        s: Session.
        u: The understood question.

    Returns:
        Queries.
    """
    prompt = load_prompt("plan").format(question=u.standalone, intent=u.intent)

    def check(p: Plan) -> list[str]:
        return [f"query {i} is longer than 20 words" for i, q in enumerate(p.queries, 1)
                if len(q.split()) > 20]  # fmt: skip

    return structured_chat(s.llm.backend, s.llm.request(prompt), Plan, check=check).queries


def retrieve(s: AskSession, query: str, tiers: list[int], exclude: list[str]) -> list[Hit]:
    """Hybrid search, skipping chunks already judged relevant.

    Args:
        s: Session.
        query: Search query.
        tiers: Tiers to search.
        exclude: Chunk ids to leave out.

    Returns:
        Up to ``top_k`` hits.
    """
    k = s.config.ask.top_k
    hits = s.index.search(query, s.embedder.query(query), tiers, k + len(exclude), "hybrid")
    return [h for h in hits if h.chunk.id not in exclude][:k]


def grade(s: AskSession, query: str, question: str, candidates: list[str]) -> Grade:
    """The judge model decides which candidates help and whether they suffice.

    Args:
        s: Session.
        query: The search query.
        question: The whole question (for context).
        candidates: Chunk ids to grade.

    Returns:
        The grade.
    """
    prompt = load_prompt("grade").format(
        question=question, query=query, evidence=format_evidence(s.index, candidates)
    )

    def check(g: Grade) -> list[str]:
        unknown = sorted(set(g.relevant) - set(candidates))
        return [f"relevant lists ids that were not shown: {unknown}"] if unknown else []

    s.switcher.use(s.judge.model)
    return structured_chat(s.judge.backend, s.judge.request(prompt), Grade, check=check)


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
    prompt = load_prompt("rewrite_query").format(
        query=query, missing=missing or "(not stated)", tried="\n".join(f"- {q}" for q in tried)
    )

    def check(r: Rewrite) -> list[str]:
        if r.query.strip().lower() in {t.strip().lower() for t in tried}:
            return ["that query was already tried; phrase it differently"]
        return ["at most 20 words"] if len(r.query.split()) > 20 else []

    s.switcher.use(s.llm.model)
    return structured_chat(s.llm.backend, s.llm.request(prompt), Rewrite, check=check).query


def widen(tiers: list[int]) -> list[int]:
    """The next tiers to search when the evidence is not enough.

    Args:
        tiers: Tiers searched so far.

    Returns:
        Tier 1 widens to tiers 1–2; wider searches stay as they are.
    """
    return [1, 2] if tiers == [1] else tiers


def find_conflicts(s: AskSession, chunk_ids: list[str]) -> list[Conflict]:
    """Values the dissertation and another source state differently.

    Only runs when the evidence mixes the dissertation with other sources. A conflict is
    kept only if both values are written in the chunks it cites.

    Args:
        s: Session.
        chunk_ids: Evidence chunks.

    Returns:
        Verified conflicts.
    """
    tiers = {s.index.chunk(c).tier for c in chunk_ids}
    if 1 not in tiers or tiers == {1}:
        return []
    prompt = load_prompt("conflicts").format(evidence=format_evidence(s.index, chunk_ids, False))
    known = set(chunk_ids)

    def check(out: Conflicts) -> list[str]:
        problems = []
        for c in out.items:
            for cid, value in ((c.dissertation_chunk, c.dissertation_value),
                               (c.other_chunk, c.other_value)):  # fmt: skip
                if cid not in known:
                    problems.append(f"{c.topic}: unknown chunk id {cid}")
                elif numbers_in(value) - numbers_in(s.index.chunk(cid).text):
                    problems.append(f"{c.topic}: {value!r} is not written in {cid}")
            if c.dissertation_chunk in known and s.index.chunk(c.dissertation_chunk).tier != 1:
                problems.append(f"{c.topic}: dissertation_chunk must be a tier-1 chunk")
        return problems

    s.switcher.use(s.judge.model)
    out = structured_chat(s.judge.backend, s.judge.request(prompt), Conflicts, check=check)
    return [c for c in out.items if c.dissertation_value != c.other_value]


def write_answer(
    s: AskSession, question: str, chunk_ids: list[str], conflicts: list[Conflict]
) -> AnswerDraft:
    """The writer answers from the evidence, every sentence citing its chunks.

    Args:
        s: Session.
        question: The self-contained question.
        chunk_ids: Evidence chunks (tier 1 first).
        conflicts: Disagreements to resolve in favour of the dissertation.

    Returns:
        The draft.
    """
    notes = "\n".join(
        f"- {c.topic}: the dissertation says {c.dissertation_value} [{c.dissertation_chunk}], "
        f"the other source says {c.other_value} [{c.other_chunk}]"
        for c in conflicts
    )
    prompt = load_prompt("answer").format(
        question=question,
        evidence=format_evidence(s.index, chunk_ids),
        conflicts=notes or "(none)",
    )
    known = set(chunk_ids)

    def check(d: AnswerDraft) -> list[str]:
        problems = []
        for i, sentence in enumerate(d.sentences, start=1):
            unknown = sorted(set(sentence.chunk_ids) - known)
            if unknown:
                problems.append(f"sentence {i} cites ids that are not in the evidence: {unknown}")
            if len(sentence.text.split()) > MAX_SENTENCE_WORDS:
                problems.append(f"sentence {i} has more than {MAX_SENTENCE_WORDS} words")
            if "[" in sentence.text:
                problems.append(f"sentence {i} has ids in its text; list them in chunk_ids")
        return problems

    s.switcher.use(s.llm.model)
    return structured_chat(s.llm.backend, s.llm.request(prompt), AnswerDraft, check=check)


def compose(
    s: AskSession,
    question: str,
    sentences: list[Sentence],
    conflicts: list[Conflict],
    dropped: int = 0,
    agent: str = "graph",
) -> Answer:
    """Number the citations and assemble the answer text.

    Args:
        s: Session.
        question: The question as asked.
        sentences: Verified sentences.
        conflicts: Resolved conflicts.
        dropped: Sentences the fact-check removed.
        agent: Which agent answered.

    Returns:
        The answer (an abstention if no sentence survived).
    """
    if not sentences:
        return Answer(question=question, text=NOT_FOUND, abstained=True,
                      reason="no supported sentence", dropped=dropped, agent=agent)  # fmt: skip
    numbers: dict[str, int] = {}
    parts = []
    for sentence in sentences:
        marks = []
        for cid in sentence.chunk_ids:
            numbers.setdefault(cid, len(numbers) + 1)
            marks.append(f"[{numbers[cid]}]")
        parts.append(f"{sentence.text.rstrip()} {''.join(marks)}")
    citations = []
    for cid, n in numbers.items():
        chunk = s.index.chunk(cid)
        citations.append(
            Citation(n=n, chunk_id=cid, source_id=chunk.source_id, label=label(s.index, cid),
                     tier=chunk.tier, text=chunk.text)
        )  # fmt: skip
    return Answer(
        question=question,
        text=" ".join(parts),
        sentences=sentences,
        citations=citations,
        conflicts=conflicts,
        dropped=dropped,
        agent=agent,
    )


def evidence_ids(s: AskSession, findings: list[Finding]) -> list[str]:
    """All relevant chunks of a turn, dissertation first, then in finding order.

    Args:
        s: Session.
        findings: The research results.

    Returns:
        Unique chunk ids.
    """
    ids = list(dict.fromkeys(c for f in findings for c in f.chunk_ids))
    return sorted(ids, key=lambda c: s.index.chunk(c).tier)


# --- graphs ---------------------------------------------------------------------------------


def build_research(s: AskSession) -> CompiledStateGraph:
    """The per-query research loop: retrieve ─▶ grade ─▶ (rewrite, widen ─▶ retrieve)*.

    Args:
        s: Session.

    Returns:
        The compiled subgraph.
    """
    max_loops = s.config.ask.max_loops

    def retrieve_node(state: ResearchState) -> ResearchState:
        with s.tracer.span("step.ask.retrieve", query=state["query"], tiers=state["tiers"]):
            hits = retrieve(s, state["query"], state["tiers"], state.get("relevant", []))
        _emit("retrieved", query=state["query"], tiers=state["tiers"],
              hits=[{"id": h.chunk.id, "score": round(h.score, 4), "tier": h.chunk.tier,
                     "text": h.chunk.text[:240]} for h in hits])  # fmt: skip
        return {"candidates": [h.chunk.id for h in hits]}

    def grade_node(state: ResearchState) -> ResearchState:
        with s.tracer.span("step.ask.grade", query=state["query"]) as span:
            g = grade(s, state["query"], state["planned"], state["candidates"])
            span.update(relevant=len(g.relevant), sufficient=g.sufficient)
        _emit("graded", query=state["query"], relevant=g.relevant, sufficient=g.sufficient,
              missing=g.missing)  # fmt: skip
        relevant = list(dict.fromkeys([*state.get("relevant", []), *g.relevant]))
        return {"relevant": relevant, "sufficient": g.sufficient, "missing": g.missing,
                "loop": state.get("loop", 0) + 1}  # fmt: skip

    def decide(state: ResearchState) -> str:
        done = state["sufficient"] or state["loop"] >= max_loops
        return "finish" if done else "rewrite"

    def rewrite_node(state: ResearchState) -> ResearchState:
        tried = [*state.get("tried", []), state["query"]]
        with s.tracer.span("step.ask.rewrite", query=state["query"]):
            query = rewrite_query(s, state["query"], state.get("missing", ""), tried)
        tiers = widen(state["tiers"])
        _emit("rewritten", query=query, tiers=tiers)
        return {"query": query, "tried": tried, "tiers": tiers}

    def finish(state: ResearchState) -> ResearchState:
        finding = Finding(
            query=state["planned"],
            queries=[*state.get("tried", []), state["query"]],
            chunk_ids=state.get("relevant", []),
            tiers=state["tiers"],
            sufficient=state.get("sufficient", False),
            loops=state.get("loop", 0),
        )
        return {"finding": finding}

    g = StateGraph(ResearchState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("finish", finish)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", decide, ["rewrite", "finish"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("finish", END)
    return g.compile()


def serializer() -> JsonPlusSerializer:
    """Checkpoint serializer that may rebuild labmate's models (and no others).

    Returns:
        The serializer.
    """
    models = [
        v
        for module in (schemas, core_schemas)
        for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, BaseModel) and v.__module__ == module.__name__
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=[*models, FactCheckLoop])


def build_graph(s: AskSession) -> StateGraph:
    """Wire the ask agent (not compiled; see :func:`compile_graph`).

    Args:
        s: Session (closed over by the nodes; not part of the checkpointed state).

    Returns:
        The graph builder.
    """
    research = build_research(s)
    verifier = build_verifier(s.llm, s.judge, s.switcher, max_rounds=1, workers=s.workers)

    def understand_node(state: AskState) -> AskState:
        with s.tracer.span("step.ask.understand") as span:
            u = understand(s, state["question"], state.get("history", []))
            span.update(intent=u.intent, options=len(u.options))
        _emit("understood", standalone=u.standalone, intent=u.intent, options=u.options)
        return {"understanding": u, "findings": None, "conflicts": [], "dropped": 0}

    def route(state: AskState) -> str:
        u = state["understanding"]
        if u.intent == "off_topic":
            return "abstain"
        return "clarify" if u.options else "plan"

    def clarify(state: AskState) -> AskState:
        u = state["understanding"]
        choice = interrupt({"question": u.standalone, "options": u.options})
        return {"understanding": u.model_copy(
            update={"standalone": f"{u.standalone} ({choice})", "options": []})}  # fmt: skip

    def plan_node(state: AskState) -> AskState:
        with s.tracer.span("step.ask.plan") as span:
            queries = plan_queries(s, state["understanding"])
            span.update(queries=queries)
        _emit("planned", queries=queries)
        return {"queries": queries}

    def fan_out(state: AskState) -> list[Send]:
        u = state["understanding"]
        tiers = START_TIERS.get(u.intent, [1])
        return [
            Send("research", {"planned": q, "query": q, "intent": u.intent, "tiers": tiers})
            for q in state["queries"]
        ]

    def research_node(task: ResearchState, config: RunnableConfig) -> AskState:
        with s.tracer.span("step.ask.research", query=task["planned"]):
            out = research.invoke(task, config)
        return {"findings": [out["finding"]]}

    def conflicts_node(state: AskState) -> AskState:
        ids = evidence_ids(s, state.get("findings") or [])
        with s.tracer.span("step.ask.conflicts", evidence=len(ids)) as span:
            found = find_conflicts(s, ids) if ids else []
            span.update(conflicts=len(found))
        _emit("conflicts", items=[c.model_dump() for c in found])
        return {"conflicts": found}

    def has_evidence(state: AskState) -> str:
        return "answer" if evidence_ids(s, state.get("findings") or []) else "abstain"

    def answer_node(state: AskState) -> AskState:
        ids = evidence_ids(s, state.get("findings") or [])
        with s.tracer.span("step.ask.answer", evidence=len(ids)):
            draft = write_answer(s, state["understanding"].standalone, ids, state["conflicts"])
        _emit("drafted", sentences=[x.text for x in draft.sentences])
        return {"draft": draft}

    def verify_node(state: AskState, config: RunnableConfig) -> AskState:
        ids = evidence_ids(s, state.get("findings") or [])
        with s.tracer.span("step.ask.verify") as span:
            out = verifier.invoke(
                {"sentences": state["draft"].sentences, "by_id": evidence_cards(s.index, ids)},
                config,
            )
            dropped = len(state["draft"].sentences) - len(out["sentences"])
            span.update(kept=len(out["sentences"]), dropped=dropped)
        _emit("verified", kept=len(out["sentences"]), dropped=dropped)
        return {"sentences": out["sentences"], "dropped": dropped}

    def finalize(state: AskState) -> AskState:
        answer = compose(
            s, state["question"], state["sentences"], state["conflicts"], state["dropped"]
        )
        return {"answer": answer, "history": [Turn(question=state["question"], answer=answer.text)]}

    def abstain(state: AskState) -> AskState:
        off = state["understanding"].intent == "off_topic"
        answer = Answer(
            question=state["question"],
            text=OFF_TOPIC if off else NOT_FOUND,
            abstained=True,
            reason="off topic" if off else "no relevant evidence",
        )
        return {"answer": answer, "history": [Turn(question=state["question"], answer=answer.text)]}

    g = StateGraph(AskState)
    for name, fn in [
        ("understand", understand_node),
        ("clarify", clarify),
        ("plan", plan_node),
        ("research", research_node),
        ("conflicts", conflicts_node),
        ("answer", answer_node),
        ("verify", verify_node),
        ("finalize", finalize),
        ("abstain", abstain),
    ]:
        g.add_node(name, fn)  # type: ignore[arg-type,call-overload,unused-ignore]
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", route, ["abstain", "clarify", "plan"])
    g.add_edge("clarify", "plan")
    g.add_conditional_edges("plan", fan_out, ["research"])
    g.add_edge("research", "conflicts")
    g.add_conditional_edges("conflicts", has_evidence, ["answer", "abstain"])
    g.add_edge("answer", "verify")
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("abstain", END)
    return g


def compile_graph(s: AskSession, threads: Path | None = None) -> CompiledStateGraph:
    """Compile the agent with a SQLite checkpointer (conversation memory).

    Args:
        s: Session.
        threads: Checkpoint database (default: ``[ask].threads``).

    Returns:
        The compiled graph.
    """
    path = threads or s.config.ask.threads
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    s.connections.append(connection)
    saver = SqliteSaver(connection, serde=serializer())
    return build_graph(s).compile(checkpointer=saver)


Chooser = Callable[[str, list[str]], str]
"""Picks one reading of an ambiguous question: ``(question, options) -> option``."""


def ask(
    s: AskSession,
    graph: CompiledStateGraph,
    question: str,
    thread: str = "default",
    choose: Chooser | None = None,
) -> Answer:
    """Ask one question in a conversation thread.

    Args:
        s: Session (its tracer records the turn).
        graph: The compiled agent.
        question: The question.
        thread: Conversation id; follow-ups on the same thread see earlier turns.
        choose: Resolves a clarification interrupt (default: the first option).

    Returns:
        The answer.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread}}
    with s.tracer.span("run", command="ask", agent="graph", thread=thread) as root:
        result = graph.invoke({"question": question}, config)
        while result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            options = list(payload["options"])
            choice = choose(payload["question"], options) if choose else options[0]
            root.update(clarified=choice)
            result = graph.invoke(Command(resume=choice), config)
        answer: Answer = result["answer"]
        root.update(abstained=answer.abstained, citations=len(answer.citations))
    return answer


INTENTS: tuple[Intent, ...] = ("thesis", "own_work", "related", "off_topic")
"""All intents (for the dashboard and tests)."""
