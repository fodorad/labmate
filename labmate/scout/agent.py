"""The scout agent and its tools."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from labmate.config import Config
from labmate.core.chat import chat_model
from labmate.core.ingest import (
    USER_AGENT,
    IngestError,
    fetch_metadata,
    parse_arxiv_id,
    search_arxiv,
)
from labmate.paper2flow.chain import ARTIFACTS, paper2flow
from labmate.paper2flow.schemas import FactChecked
from labmate.scout.prompts import load_prompt

log = logging.getLogger(__name__)

RECURSION_LIMIT = 40
"""Graph steps the agent may take (each tool call is two)."""

_ID = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")


def cited_ids(notes: str) -> set[str]:
    """The arXiv ids a text cites.

    Args:
        notes: Markdown notes.

    Returns:
        Bare ids (no version).
    """
    return set(_ID.findall(notes))


def overview_text(
    config: Config,
    arxiv_id: str,
    http: httpx.BaseTransport | None,
    ollama: httpx.BaseTransport | None,
) -> str:
    """Run paper2flow on a paper and return its fact-checked cards as text.

    Args:
        config: Loaded configuration.
        arxiv_id: The paper.
        http: Web transport (the network if omitted; tests).
        ollama: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        The four cards, one bullet per line, and where the PDF is.
    """
    pdf = paper2flow(config, arxiv_id, web=http, ollama=ollama)
    checked = FactChecked.model_validate_json((pdf.parent / ARTIFACTS["checked"]).read_text())
    lines = [
        f"{card.title}: " + " ".join(b.text for b in card.bullets) for card in checked.cards.cards
    ]
    return "\n".join([*lines, f"(overview: {pdf})"])


def run_scout(
    config: Config,
    topic: str,
    out_dir: Path,
    max_deep: int = 2,
    http: httpx.BaseTransport | None = None,
    ollama: httpx.BaseTransport | None = None,
) -> Path | None:
    """Scout a topic: the agent searches, reads, and writes ``notes.md``.

    Args:
        config: Loaded configuration.
        topic: What to look into.
        out_dir: Where ``notes.md`` goes.
        max_deep: Most papers the agent may run the slow deep look on.
        http: Web transport (the network if omitted; tests).
        ollama: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        The notes file, or ``None`` if the agent stopped without writing notes.
    """
    web = httpx.Client(
        transport=http or httpx.HTTPTransport(retries=2),
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    seen: dict[str, str] = {}  # arXiv id -> title, from searches
    read: set[str] = set()
    deep: list[str] = []
    notes_file = out_dir / "notes.md"

    @tool("search_arxiv")
    def search(query: str) -> str:
        """Search arXiv and list the best papers with their ids.

        Args:
            query: Search terms.
        """
        hits = search_arxiv(query, web)
        seen.update({i: m.title for i, m in hits})
        return (
            "\n".join(
                f"{i}: {m.title} ({m.published[:4]}, {', '.join(m.authors[:2])})" for i, m in hits
            )
            or "no results"
        )

    @tool("read_abstract")
    def read_abstract(arxiv_id: str) -> str:
        """Read a paper's title and abstract.

        Args:
            arxiv_id: An arXiv id such as 1706.03762.
        """
        try:
            bare = parse_arxiv_id(arxiv_id)
            meta = fetch_metadata(bare, web)
        except IngestError as e:
            return f"cannot read {arxiv_id}: {e}"
        read.add(bare)
        return f"{meta.title} ({meta.published[:4]})\n{meta.abstract}"

    @tool("read_overview")
    def read_overview(arxiv_id: str) -> str:
        """A deep, checked look at one paper (slow; limited).

        Args:
            arxiv_id: An arXiv id such as 1706.03762.
        """
        try:
            bare = parse_arxiv_id(arxiv_id)
        except IngestError as e:
            return f"cannot read {arxiv_id}: {e}"
        if bare not in deep and len(deep) >= max_deep:
            return f"deep look limit reached ({max_deep}); use read_abstract instead"
        result = overview_text(config, bare, http, ollama)
        if bare not in deep:
            deep.append(bare)
        read.add(bare)
        return result

    @tool("write_notes")
    def write_notes(notes: str) -> str:
        """Finish: save the notes (Markdown) after checking their citations.

        Args:
            notes: The notes; every paper cited by its arXiv id in brackets.
        """
        cited = cited_ids(notes)
        if not cited:
            return "cite at least one paper by its arXiv id, like [1706.03762]"
        if unread := sorted(cited - read):
            return f"you have not read {unread}; read them first or remove them from the notes"
        out_dir.mkdir(parents=True, exist_ok=True)
        notes_file.write_text(notes.rstrip() + "\n")
        return "saved"

    tools: list[BaseTool] = [search, read_abstract, read_overview, write_notes]
    agent = create_agent(
        model=chat_model(config, config.models.text, transport=ollama),
        tools=tools,
        system_prompt=load_prompt("scout").format(topic=topic, deep=max_deep),
    )
    config_: RunnableConfig = {"recursion_limit": RECURSION_LIMIT, "run_name": "scout"}
    try:
        agent.invoke({"messages": [HumanMessage(topic)]}, config_)
    finally:
        web.close()
    log.info("scout read %d paper(s), %d deeply", len(read), len(deep))
    return notes_file if notes_file.exists() else None
