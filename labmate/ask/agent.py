"""The ask agent: a tool-calling (ReAct) agent over the library.

The same task as :mod:`labmate.ask.graph`, but the model decides everything: which tool to
call, how to rephrase a search, when to read more context and when to stop. Nothing fixes
the order of steps, only the tools and the step limit. What the model writes is still
checked by code: a sentence stays only if the chunks it cites exist and contain its numbers
(see :func:`labmate.ask.answer.compose`).
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from labmate.ask.answer import compose
from labmate.ask.evidence import context, label
from labmate.ask.prompts import load_prompt
from labmate.ask.schemas import Answer, Sentence
from labmate.ask.session import AskSession
from labmate.ask.verify import verify_sentences

RECURSION_LIMIT = 16
"""Graph steps the agent may take (each tool call is two)."""

_CITE = re.compile(r"\[([a-z0-9][a-z0-9-]*:\d{4})\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?])")
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def make_tools(s: AskSession) -> list[BaseTool]:
    """The library tools, bound to a session.

    Args:
        s: Session.

    Returns:
        ``search_library``, ``read_context`` and ``list_sources``.
    """

    @tool
    def search_library(query: str) -> str:
        """Search the library and return the best passages with their chunk ids.

        Args:
            query: What to look for.
        """
        hits = s.index.search(query, s.embedder.query(query), s.config.ask.top_k)
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
        """List the documents in the library."""
        return "\n".join(f"{x.id}: {x.name}. {x.title}" for x in s.index.sources())

    return [search_library, read_context, list_sources]


def build_agent(s: AskSession) -> Any:
    """The agent over the writer model.

    Args:
        s: Session.

    Returns:
        A compiled LangGraph graph (``create_agent`` builds one).
    """
    return create_agent(model=s.writer, tools=make_tools(s), system_prompt=load_prompt("agent"))


def parse_answer(s: AskSession, question: str, text: str) -> Answer:
    """Turn the agent's free text into an :class:`Answer`.

    Sentences are split off the text with their ``[chunk id]`` citations. A sentence without a
    valid citation is dropped, and so is one the judge model or the number check finds
    unsupported by the chunks it cites.

    Args:
        s: Session.
        question: The question.
        text: The agent's final message.

    Returns:
        The answer; an abstention carries the agent's own text when nothing was cited.
    """
    sentences, uncited = [], 0
    for raw in _SENTENCES.split(text.strip()):
        ids = []
        for cid in _CITE.findall(raw):
            try:
                s.index.chunk(cid)
                ids.append(cid)
            except KeyError:
                continue
        clean = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", _CITE.sub("", raw)).strip()
        if ids and clean:
            sentences.append(Sentence(text=" ".join(clean.split()), chunk_ids=ids))
        elif clean:
            uncited += 1
    kept, rejected = verify_sentences(s, question, sentences)
    answer = compose(s.index, question, kept, dropped=uncited + rejected, agent="agent")
    if answer.abstained:
        return answer.model_copy(update={"text": text.strip() or answer.text})
    return answer


def ask_agent(s: AskSession, agent: Any, question: str) -> Answer:
    """Ask the agent one question.

    Args:
        s: Session.
        agent: From :func:`build_agent`.
        question: The question.

    Returns:
        The parsed answer.
    """
    result = agent.invoke(
        {"messages": [HumanMessage(question)]}, {"recursion_limit": RECURSION_LIMIT}
    )
    final = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
    text = final.content if final is not None and isinstance(final.content, str) else ""
    return parse_answer(s, question, text)
