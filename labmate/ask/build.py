"""Build the index: sections, chunks and embeddings.

Per source (skipped if already indexed, unless ``fresh``)::

    PDF ─▶ section tree ─▶ chunks (text, captions) ─▶ embeddings ─▶ index

The only model used is the embedding model.
"""

from __future__ import annotations

import logging

from labmate.ask.chunk import chunk_document, outline_sections
from labmate.ask.library import Library, Source
from labmate.ask.session import AskSession

log = logging.getLogger(__name__)


def index_source(s: AskSession, library: Library, source: Source) -> int:
    """Index one source.

    Args:
        s: Session.
        library: The library (file locations).
        source: The source.

    Returns:
        The number of chunks stored.
    """
    sections = outline_sections(library.path(source), source.id)
    chunks = chunk_document(sections, source, s.config.ask.chunk_words)
    vectors = s.embedder.documents([c.text for c in chunks])
    s.index.add_source(source, sections, chunks, vectors)
    log.info("indexed %s: %d sections, %d chunks", source.id, len(sections), len(chunks))
    return len(chunks)


def build_index(s: AskSession, library: Library, fresh: bool = False) -> dict[str, int]:
    """Index every source of the library that is not indexed yet.

    An index built with another embedding model or chunk size is rebuilt completely:
    its vectors and chunks can't be mixed with new ones.

    Args:
        s: Session.
        library: The library.
        fresh: Re-index sources that are already in the index.

    Returns:
        Source id to chunk count for the sources indexed now.
    """
    settings = {"embed_model": s.embedder.model, "chunk_words": str(s.config.ask.chunk_words)}
    built_with = s.index.meta()
    changed = sorted(k for k, v in settings.items() if k in built_with and built_with[k] != v)
    if changed and not fresh:
        log.warning("index settings changed (%s): re-indexing every source", ", ".join(changed))
        fresh = True
    done: dict[str, int] = {}
    for source in library.sources:
        if s.index.has_source(source.id) and not fresh:
            log.info("  reuse %s", source.id)
            continue
        done[source.id] = index_source(s, library, source)
    s.index.set_meta(**settings)
    return done
