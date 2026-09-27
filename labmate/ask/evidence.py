"""Evidence as the models and the reader see it: context windows and citation labels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from labmate.ask.chunk import section_number
from labmate.ask.index import Index
from labmate.ask.library import TIER_NAMES
from labmate.core.schemas import ClaimCard

CONTEXT_WINDOW = 1
"""Neighbouring chunks shown with a hit (each side, same section)."""


def label(index: Index, chunk_id: str) -> str:
    """How a chunk is cited: ``Dissertation §4.2, p. 57``.

    Args:
        index: The index.
        chunk_id: Chunk id.

    Returns:
        The citation label.
    """
    chunk = index.chunk(chunk_id)
    source = index.source(chunk.source_id)
    section = index.section(chunk.section_id)
    where = "thesis " + chunk.thesis if chunk.kind == "thesis" else section_number(section)
    return f"{source.name} {where}, p. {chunk.page}"


def context(index: Index, chunk_id: str) -> str:
    """The text a model reads for a chunk: the chunk with its neighbours in the section.

    Args:
        index: The index.
        chunk_id: Chunk id.

    Returns:
        Whitespace-joined text.
    """
    return " ".join(c.text for c in index.neighbors(chunk_id, CONTEXT_WINDOW))


def format_evidence(index: Index, chunk_ids: Iterable[str], expand: bool = True) -> str:
    """Evidence block for a prompt: id, citation, tier and text per chunk.

    Args:
        index: The index.
        chunk_ids: Chunks, in the order to show them.
        expand: Show each chunk with its neighbours.

    Returns:
        The block (``(none)`` if empty).
    """
    lines = []
    for cid in dict.fromkeys(chunk_ids):
        chunk = index.chunk(cid)
        text = context(index, cid) if expand else chunk.text
        tier = TIER_NAMES.get(chunk.tier, str(chunk.tier))
        lines.append(f"[{cid}] {label(index, cid)} (tier {chunk.tier}, {tier}):\n{text}")
    return "\n\n".join(lines) or "(none)"


def evidence_cards(index: Index, chunk_ids: Sequence[str]) -> dict[str, ClaimCard]:
    """Chunks as claim cards, so the shared fact-check loop can verify answer sentences.

    Args:
        index: The index.
        chunk_ids: Evidence chunks.

    Returns:
        Chunk id to a card whose "quote" is the chunk's context window.
    """
    cards = {}
    for cid in dict.fromkeys(chunk_ids):
        chunk = index.chunk(cid)
        cards[cid] = ClaimCard(
            id=cid,
            claim=label(index, cid),
            evidence_quote=context(index, cid),
            kind="method",
            section=index.section(chunk.section_id).title,
            page=chunk.page,
            match=100.0,
        )
    return cards
