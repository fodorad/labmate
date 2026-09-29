"""Baseline: LangChain's prebuilt tool-calling agent (``create_agent``) with library tools.

The same question-answering task as :mod:`labmate.ask.graph`, but the model decides
everything in one ReAct loop: which tool to call, when to widen the search, when to
stop. It has the same tools and the same recorded backend, so the evaluation compares
the control flow, not the plumbing.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from labmate.ask.evidence import context, label
from labmate.ask.graph import compose
from labmate.ask.prompts import load_prompt
from labmate.ask.schemas import Answer, Sentence
from labmate.ask.session import AskSession
from labmate.core.lc import RecordedChatModel

SCOPES = {"dissertation": [1], "own": [1, 2], "all": [1, 2, 3]}
"""Search scopes offered to the agent, as tiers."""

RECURSION_LIMIT = 16
"""Graph steps the agent may take (each tool call is two)."""

_CITE = re.compile(r"\[([a-z0-9][a-z0-9-]*:\d{4})\]")
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def make_tools(s: AskSession) -> list[BaseTool]:
    """The library tools, bound to a session.

    Args:
        s: Session.

    Returns:
        ``search_library``, ``read_context`` and ``list_sources``.
    """

    @tool
    def search_library(
        query: str, scope: Literal["dissertation", "own", "all"] = "dissertation"
    ) -> str:
        """Search the library and return the best passages with their chunk ids.

        Args:
            query: What to look for.
            scope: "dissertation" (the source of truth), "own" (dissertation and papers) or
                "all" (also outside papers).
        """
        tiers = SCOPES.get(scope, [1])
        hits = s.index.search(query, s.embedder.query(query), tiers, s.config.ask.top_k)
        return "\n\n".join(f"[{h.chunk.id}] {label(s.index, h.chunk.id)}: {h.chunk.text}"
                           for h in hits) or "no results"  # fmt: skip

    @tool
    def read_context(chunk_id: str) -> str:
        """Read a passage together with the passages around it.

        Args:
            chunk_id: A chunk id returned by search_library, e.g. "dissertation:0042".
        """
        try:
            return f"[{chunk_id}] {label(s.index, chunk_id)}: {context(s.index, chunk_id)}"
        except KeyError:
            return f"unknown chunk id {chunk_id!r}"

    @tool
    def list_sources() -> str:
        """List the documents in the library with their tiers."""
        return "\n".join(f"{x.id}: {x.name} (tier {x.tier}) {x.title}" for x in s.index.sources())

    return [search_library, read_context, list_sources]


def build_agent(s: AskSession) -> Any:
    """The prebuilt agent over the recorded chat model.

    Args:
        s: Session.

    Returns:
        A compiled LangGraph graph (``create_agent`` builds one).
    """
    model = RecordedChatModel(llm=s.llm)
    return create_agent(model=model, tools=make_tools(s), system_prompt=load_prompt("agent"))


def parse_answer(s: AskSession, question: str, text: str) -> Answer:
    """Turn the agent's free text into an :class:`Answer` (sentences and their citations).

    Sentences without a valid citation are kept out of the cited sentences, so the
    answer evaluation scores both agents the same way.

    Args:
        s: Session.
        question: The question.
        text: The agent's final message.

    Returns:
        The answer.
    """
    sentences = []
    for raw in _SENTENCES.split(text.strip()):
        ids = []
        for cid in _CITE.findall(raw):
            try:
                s.index.chunk(cid)
                ids.append(cid)
            except KeyError:
                continue
        clean = _CITE.sub("", raw).strip()
        if ids and clean:
            sentences.append(Sentence(text=" ".join(clean.split()), chunk_ids=ids))
    answer = compose(s, question, sentences, [], agent="prebuilt")
    if answer.abstained:
        return answer.model_copy(update={"text": text.strip() or answer.text})
    return answer


def ask_prebuilt(s: AskSession, agent: Any, question: str) -> Answer:
    """Ask the prebuilt agent one question.

    Args:
        s: Session.
        agent: From :func:`build_agent`.
        question: The question.

    Returns:
        The parsed answer.
    """
    with s.tracer.span("run", command="ask", agent="prebuilt") as root:
        s.switcher.use(s.llm.model)
        result = agent.invoke(
            {"messages": [HumanMessage(question)]}, {"recursion_limit": RECURSION_LIMIT}
        )
        final = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        text = final.content if final is not None and isinstance(final.content, str) else ""
        tool_calls = sum(len(m.tool_calls) for m in result["messages"] if isinstance(m, AIMessage))
        answer = parse_answer(s, question, text)
        root.update(tool_calls=tool_calls, abstained=answer.abstained)
    return answer
