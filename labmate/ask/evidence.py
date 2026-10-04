"""Evidence as the models and the reader see it: context windows and citation labels."""

from __future__ import annotations

from collections.abc import Iterable

from labmate.ask.chunk import section_number
from labmate.ask.index import Index

CONTEXT_WINDOW = 1
"""Neighbouring chunks shown with a hit (each side, same section)."""

GRADE_WINDOW = 0
"""Neighbours shown when grading: relevance can be told from the matched chunk alone, and the
neighbours would make the grading prompt three times longer."""


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
    return f"{source.name} {section_number(section)}, p. {chunk.page}"


def context(index: Index, chunk_id: str, window: int = CONTEXT_WINDOW) -> str:
    """The text a model reads for a chunk: the chunk with its neighbours in the section.

    Args:
        index: The index.
        chunk_id: Chunk id.
        window: Neighbouring chunks on each side.

    Returns:
        Whitespace-joined text.
    """
    return " ".join(c.text for c in index.neighbors(chunk_id, window))


def format_evidence(index: Index, chunk_ids: Iterable[str], window: int = CONTEXT_WINDOW) -> str:
    """Evidence block for a prompt: id, citation and text per chunk.

    Args:
        index: The index.
        chunk_ids: Chunks, in the order to show them.
        window: Neighbouring chunks shown with each, on each side.

    Returns:
        The block (``(none)`` if empty).
    """
    lines = [
        f"[{cid}] {label(index, cid)}:\n{context(index, cid, window)}"
        for cid in dict.fromkeys(chunk_ids)
    ]
    return "\n\n".join(lines) or "(none)"
