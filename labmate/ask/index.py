"""The search index: one SQLite file with sections, chunks, a BM25 index and vectors.

No vector database: a PhD's worth of documents is a few thousand chunks, and exact
cosine search over a few thousand vectors with numpy takes well under a millisecond, so
an approximate index would only add a dependency and recall loss. SQLite's FTS5 gives
BM25; the two rankings are fused with reciprocal rank fusion (RRF).
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from labmate.ask.chunk import Chunk, DocSection
from labmate.ask.library import Source

RRF_K = 60
"""Reciprocal rank fusion constant (the usual value from Cormack et al., 2009)."""

CANDIDATES = 30
"""Hits taken from each ranking before fusion."""

_STOP = frozenset(
    "a an and are as at be by does do for from how in is it its of on or that the this to "
    "was what when where which who why with were has have had i my me your you".split()
)

SCHEMA = """
create table if not exists meta(key text primary key, value text);
create table if not exists sources(
    id text primary key, label text, title text, year int, venue text);
create table if not exists sections(
    id text primary key, source_id text, level int, number text, title text, path text,
    page int, text text);
create table if not exists chunks(
    rowid integer primary key, id text unique, source_id text, section_id text,
    kind text, page int, text text, vector blob);
create virtual table if not exists chunks_fts using fts5(
    text, content='chunks', content_rowid='rowid', tokenize='porter unicode61');
"""


class Hit(BaseModel):
    """A retrieved chunk.

    Attributes:
        chunk: The chunk.
        score: Score of the method that produced the hit (higher is better).
        bm25_rank: 1-based rank in the BM25 list, 0 if absent.
        dense_rank: 1-based rank in the dense list, 0 if absent.
    """

    chunk: Chunk
    score: float
    bm25_rank: int = 0
    dense_rank: int = 0


def fts_query(text: str) -> str:
    """An FTS5 query matching any content word of ``text`` (BM25 ranks the matches).

    Args:
        text: A natural-language query.

    Returns:
        The FTS5 expression, or ``""`` if the text has no content word.
    """
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in _STOP and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: sum of ``1 / (k + rank)`` over the rankings.

    Args:
        rankings: Lists of ids, best first.
        k: Damping constant.

    Returns:
        ``(id, score)`` pairs, best first (ties keep first-seen order).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _locked[F: Callable[..., Any]](method: F) -> F:
    """Run a method under the index's lock (parallel graph branches share one connection)."""

    @wraps(method)
    def wrapper(self: Index, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class Index:
    """Read/write access to the index file.

    Args:
        path: SQLite file (created if missing).
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        # one connection, shared by parallel graph branches: serialise its use
        self._lock = threading.RLock()
        self._matrix_cache: tuple[list[str], NDArray[np.float32]] | None = None

    def close(self) -> None:
        """Close the database."""
        self.db.close()

    # --- writing ---------------------------------------------------------------------------

    @_locked
    def set_meta(self, **values: str) -> None:
        """Store build metadata (embedding model, chunk size, ...).

        Args:
            values: Key/value pairs.
        """
        self.db.executemany(
            "insert or replace into meta values (?, ?)", [(k, str(v)) for k, v in values.items()]
        )
        self.db.commit()

    @_locked
    def meta(self) -> dict[str, str]:
        """Build metadata.

        Returns:
            Key/value pairs.
        """
        return dict(self.db.execute("select key, value from meta").fetchall())

    @_locked
    def has_source(self, source_id: str) -> bool:
        """Whether a source is already indexed.

        Args:
            source_id: Source id.

        Returns:
            True if its chunks are in the index.
        """
        row = self.db.execute("select 1 from chunks where source_id = ? limit 1", (source_id,))
        return row.fetchone() is not None

    @_locked
    def remove_source(self, source_id: str) -> None:
        """Delete everything of one source (before re-indexing it).

        Args:
            source_id: Source id.
        """
        rows = self.db.execute(
            "select rowid, text from chunks where source_id = ?", (source_id,)
        ).fetchall()
        self.db.executemany(
            "insert into chunks_fts(chunks_fts, rowid, text) values ('delete', ?, ?)", rows
        )
        for table in ("chunks", "sections"):
            self.db.execute(f"delete from {table} where source_id = ?", (source_id,))  # noqa: S608
        self.db.execute("delete from sources where id = ?", (source_id,))
        self.db.commit()
        self._matrix_cache = None

    @_locked
    def add_source(
        self,
        source: Source,
        sections: Sequence[DocSection],
        chunks: Sequence[Chunk],
        vectors: NDArray[np.float32],
    ) -> None:
        """Store a source with its sections, chunks and their vectors.

        Args:
            source: The source.
            sections: Its sections.
            chunks: Its chunks.
            vectors: One normalised vector per chunk.
        """
        self.remove_source(source.id)
        self.db.execute(
            "insert into sources values (?, ?, ?, ?, ?)",
            (source.id, source.name, source.title, source.year, source.venue),
        )
        self.db.executemany(
            "insert into sections values (?, ?, ?, ?, ?, ?, ?, ?)",
            [(s.id, s.source_id, s.level, s.number, s.title, " › ".join(s.path), s.page, s.text)
             for s in sections],
        )  # fmt: skip
        for chunk, vector in zip(chunks, vectors, strict=True):
            cur = self.db.execute(
                "insert into chunks(id, source_id, section_id, kind, page, text, vector)"
                " values (?, ?, ?, ?, ?, ?, ?)",
                (chunk.id, chunk.source_id, chunk.section_id, chunk.kind, chunk.page,
                 chunk.text, np.asarray(vector, np.float32).tobytes()),
            )  # fmt: skip
            self.db.execute(
                "insert into chunks_fts(rowid, text) values (?, ?)", (cur.lastrowid, chunk.text)
            )
        self.db.commit()
        self._matrix_cache = None

    # --- reading ---------------------------------------------------------------------------

    def _chunk(self, row: tuple[object, ...]) -> Chunk:
        keys = ("id", "source_id", "section_id", "kind", "page", "text")
        return Chunk.model_validate(dict(zip(keys, row, strict=True)))

    _COLS = "id, source_id, section_id, kind, page, text"

    @_locked
    def chunk(self, chunk_id: str) -> Chunk:
        """Look up a chunk.

        Args:
            chunk_id: Chunk id.

        Returns:
            The chunk.

        Raises:
            KeyError: If it does not exist.
        """
        row = self.db.execute(
            f"select {self._COLS} from chunks where id = ?",  # noqa: S608
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise KeyError(chunk_id)
        return self._chunk(row)

    @_locked
    def chunks(self, source_id: str | None = None) -> list[Chunk]:
        """All chunks, optionally of one source, in id order.

        Args:
            source_id: Restrict to this source.

        Returns:
            The chunks.
        """
        where, args = ("where source_id = ?", (source_id,)) if source_id else ("", ())
        rows = self.db.execute(
            f"select {self._COLS} from chunks {where} order by rowid",  # noqa: S608
            args,
        ).fetchall()
        return [self._chunk(r) for r in rows]

    @_locked
    def neighbors(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        """A chunk with up to ``window`` text chunks before and after it in its section.

        Small-to-big retrieval: search matches small chunks, the models read the window.

        Args:
            chunk_id: Chunk id.
            window: Chunks on each side.

        Returns:
            The chunks in reading order (just the chunk itself for non-text chunks).
        """
        chunk = self.chunk(chunk_id)
        if chunk.kind != "text":
            return [chunk]
        rows = self.db.execute(
            f"select {self._COLS} from chunks where section_id = ? and kind = 'text'"  # noqa: S608
            " order by rowid",
            (chunk.section_id,),
        ).fetchall()
        chunks = [self._chunk(r) for r in rows]
        at = [c.id for c in chunks].index(chunk_id)
        return chunks[max(0, at - window) : at + window + 1]

    @_locked
    def section(self, section_id: str) -> DocSection:
        """Look up a section (for parent-document expansion and citations).

        Args:
            section_id: Section id.

        Returns:
            The section.

        Raises:
            KeyError: If it does not exist.
        """
        row = self.db.execute(
            "select id, source_id, level, number, title, path, page, text from sections"
            " where id = ?",
            (section_id,),
        ).fetchone()
        if row is None:
            raise KeyError(section_id)
        keys = ("id", "source_id", "level", "number", "title", "path", "page", "text")
        data = dict(zip(keys, row, strict=True))
        data["path"] = str(data["path"]).split(" › ")
        return DocSection.model_validate(data)

    @_locked
    def sources(self) -> list[Source]:
        """The indexed sources.

        Returns:
            Sources, ordered by id.
        """
        rows = self.db.execute(
            "select id, label, title, year, venue from sources order by id"
        ).fetchall()
        return [
            Source(id=i, file="", label=lb, title=ti, year=y, venue=v or "")
            for i, lb, ti, y, v in rows
        ]

    def source(self, source_id: str) -> Source:
        """Look up an indexed source.

        Args:
            source_id: Source id.

        Returns:
            The source.

        Raises:
            KeyError: If it is not indexed.
        """
        for source in self.sources():
            if source.id == source_id:
                return source
        raise KeyError(source_id)

    # --- search ----------------------------------------------------------------------------

    @_locked
    def bm25(self, query: str, k: int) -> list[Hit]:
        """Full-text search ranked by BM25.

        Args:
            query: Natural-language query.
            k: Maximum hits.

        Returns:
            Hits, best first.
        """
        expression = fts_query(query)
        if not expression:
            return []
        rows = self.db.execute(
            "select c.id, c.source_id, c.section_id, c.kind, c.page, c.text,"
            " bm25(chunks_fts) from chunks_fts join chunks c on c.rowid = chunks_fts.rowid"
            " where chunks_fts match ? order by bm25(chunks_fts) limit ?",
            (expression, k),
        ).fetchall()
        return [
            Hit(chunk=self._chunk(r[:6]), score=-float(r[6]), bm25_rank=i)
            for i, r in enumerate(rows, start=1)
        ]

    @_locked
    def _matrix(self) -> tuple[list[str], NDArray[np.float32]]:
        if self._matrix_cache is None:
            rows = self.db.execute("select id, vector from chunks order by rowid").fetchall()
            vectors = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
            matrix = np.vstack(vectors) if vectors else np.zeros((0, 1), np.float32)
            self._matrix_cache = ([r[0] for r in rows], matrix)
        return self._matrix_cache

    @_locked
    def dense(self, vector: NDArray[np.float32], k: int) -> list[Hit]:
        """Exact cosine search over the chunk vectors.

        Args:
            vector: Normalised query vector.
            k: Maximum hits.

        Returns:
            Hits, best first.
        """
        ids, matrix = self._matrix()
        if not ids:
            return []
        scores = matrix @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(chunk=self.chunk(ids[i]), score=float(scores[i]), dense_rank=rank)
            for rank, i in enumerate(order, start=1)
        ]

    @_locked
    def search(self, query: str, vector: NDArray[np.float32], k: int) -> list[Hit]:
        """Hybrid search: BM25 and dense rankings fused with reciprocal rank fusion.

        Args:
            query: Natural-language query (for BM25).
            vector: Its embedding (for dense search).
            k: Maximum hits.

        Returns:
            Hits, best first.
        """
        lexical = self.bm25(query, CANDIDATES)
        semantic = self.dense(vector, CANDIDATES)
        ranks = {
            "bm25": {h.chunk.id: h.bm25_rank for h in lexical},
            "dense": {h.chunk.id: h.dense_rank for h in semantic},
        }
        chunks = {h.chunk.id: h.chunk for h in [*lexical, *semantic]}
        fused = rrf([[h.chunk.id for h in lexical], [h.chunk.id for h in semantic]])[:k]
        return [
            Hit(
                chunk=chunks[cid],
                score=score,
                bm25_rank=ranks["bm25"].get(cid, 0),
                dense_rank=ranks["dense"].get(cid, 0),
            )
            for cid, score in fused
        ]
