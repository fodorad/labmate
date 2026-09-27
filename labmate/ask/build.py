"""Build the index: sections, chunks, chapter summaries, claim cards and embeddings.

Per source (skipped if already indexed, unless ``fresh``)::

    PDF ─▶ section tree ─▶ chunks (text, captions, thesis points)
        ─▶ chapter summaries (writer model; tiers 1–2)
        ─▶ claim cards (shared extraction step; tiers 1–2; the retrieval test set)
        ─▶ embeddings ─▶ index

Every model call goes through the recorded backend, so an index rebuild in replay mode
reproduces the same index without Ollama.
"""

from __future__ import annotations

import logging

from labmate.ask.chunk import Chunk, DocSection, chunk_document, outline_sections
from labmate.ask.library import Library, Source
from labmate.ask.prompts import load_prompt
from labmate.ask.session import AskSession
from labmate.core.extract import extract_claims
from labmate.core.parallel import parallel_map
from labmate.core.schemas import ClaimCard, Paper, Section

log = logging.getLogger(__name__)

SUMMARY_MIN_WORDS = 300
"""Chapters shorter than this are not summarised (their chunks already say it all)."""

SUMMARY_INPUT_WORDS = 3000
"""Words of a chapter shown to the summariser."""


def chapter_text(sections: list[DocSection], chapter: DocSection) -> str:
    """A chapter's full text: the chapter section and all sections below it.

    Args:
        sections: All sections of the document, in order.
        chapter: A level-1 section.

    Returns:
        The concatenated text.
    """
    start = sections.index(chapter)
    parts = [chapter.text]
    for section in sections[start + 1 :]:
        if section.level <= chapter.level:
            break
        parts.append(section.text)
    return "\n".join(parts)


def summarize_chapters(
    s: AskSession, source: Source, sections: list[DocSection], start_index: int
) -> list[Chunk]:
    """One summary chunk per chapter, written by the writer model.

    Summaries answer broad questions ("what is chapter 4 about?") that no single
    paragraph answers.

    Args:
        s: Session.
        source: The document.
        sections: Its sections.
        start_index: Number of the first chunk.

    Returns:
        Summary chunks.
    """
    chapters = [
        c
        for c in sections
        if c.level == 1 and len(chapter_text(sections, c).split()) >= SUMMARY_MIN_WORDS
    ]
    template = load_prompt("summary")

    def summarize(chapter: DocSection) -> str:
        text = " ".join(chapter_text(sections, chapter).split()[:SUMMARY_INPUT_WORDS])
        prompt = template.format(
            title=source.title or source.name, chapter=chapter.title, text=text
        )
        return " ".join(s.llm.backend.chat(s.llm.request(prompt)).content.split())

    s.switcher.use(s.llm.model)
    summaries = parallel_map(summarize, chapters, s.workers)
    return [
        Chunk(
            id=f"{source.id}:{start_index + i:04d}",
            source_id=source.id,
            tier=source.tier,
            section_id=chapter.id,
            kind="summary",
            page=chapter.page,
            text=f"Summary of {chapter.title}: {summary}",
        )
        for i, (chapter, summary) in enumerate(zip(chapters, summaries, strict=True))
        if summary
    ]


def source_claims(s: AskSession, source: Source, sections: list[DocSection]) -> list[ClaimCard]:
    """Verified claim cards of a document (the shared extraction step and quote guard).

    Args:
        s: Session.
        source: The document.
        sections: Its sections.

    Returns:
        Claim cards whose quotes were found in the text.
    """
    paper = Paper(
        paper_id=source.id,
        title=source.title or source.name,
        sections=[Section(title=x.title, page=x.page, text=x.text) for x in sections],
    )
    s.switcher.use(s.llm.model)
    return extract_claims(paper, s.llm, s.workers).cards


def index_source(
    s: AskSession, library: Library, source: Source, claims: bool = True
) -> tuple[int, int]:
    """Index one source.

    Args:
        s: Session.
        library: The library (file locations).
        source: The source.
        claims: Also extract claim cards (tiers 1–2).

    Returns:
        ``(chunks, claim cards)`` stored.
    """
    with s.tracer.span("step.index.source", source=source.id, tier=source.tier) as span:
        sections = outline_sections(library.path(source), source.id)
        chunks = chunk_document(sections, source, s.config.ask.chunk_words)
        cards: list[ClaimCard] = []
        if source.tier <= 2:
            chunks += summarize_chapters(s, source, sections, len(chunks))
            if claims:
                cards = source_claims(s, source, sections)
        vectors = s.embedder.documents([c.text for c in chunks])
        s.index.add_source(source, sections, chunks, vectors, cards)
        span.update(sections=len(sections), chunks=len(chunks), claims=len(cards))
    log.info("indexed %s: %d sections, %d chunks, %d claims", source.id, len(sections),
             len(chunks), len(cards))  # fmt: skip
    return len(chunks), len(cards)


def build_index(
    s: AskSession, library: Library, fresh: bool = False, claims: bool = True
) -> dict[str, tuple[int, int]]:
    """Index every source of the library that is not indexed yet.

    Args:
        s: Session.
        library: The library.
        fresh: Re-index sources that are already in the index.
        claims: Extract claim cards (needed for the retrieval evaluation).

    Returns:
        Source id to ``(chunks, claim cards)`` for the sources indexed now.
    """
    done: dict[str, tuple[int, int]] = {}
    with s.tracer.span("run", command="ask.index", sources=len(library.sources)):
        for source in library.sources:
            if s.index.has_source(source.id) and not fresh:
                log.info("  reuse %s", source.id)
                continue
            done[source.id] = index_source(s, library, source, claims)
        s.index.set_meta(embed_model=s.embedder.model, chunk_words=str(s.config.ask.chunk_words))
    return done
