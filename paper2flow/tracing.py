"""Minimal JSONL tracing: one line per finished span, nested via context variables.

A span is written when it *ends*, so ``trace.jsonl`` is append-only and survives crashes
(every completed step is on disk). Field names follow OpenTelemetry conventions
(``trace_id``, ``span_id``, ``parent_id``), so exporting to an OTel backend later is a
mapping exercise, not a redesign.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper2flow.llm.client import Backend
from paper2flow.llm.types import ChatRequest, ChatResponse, ImageRequest, ImageResponse

_current_span: ContextVar[str | None] = ContextVar("current_span", default=None)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class Tracer:
    """Collects spans in memory and optionally appends them to a JSONL file.

    Args:
        path: JSONL file to append to. ``None`` keeps spans in memory only.
        trace_id: Identifier shared by all spans of one run.
    """

    def __init__(self, path: Path | None = None, trace_id: str | None = None) -> None:
        self.path = path
        self.trace_id = trace_id or _new_id()
        self.spans: list[dict[str, Any]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        """Time a block of work as a span.

        The yielded dict can be updated inside the block to attach results (tokens,
        verdicts, ...). Exceptions are recorded with ``status="error"`` and re-raised.

        Args:
            name: Span name, e.g. ``"fact_check"``.
            **attrs: Initial attributes.

        Yields:
            The mutable attribute dict of the span.
        """
        span_id = _new_id()
        parent_id = _current_span.get()
        token = _current_span.set(span_id)
        record: dict[str, Any] = dict(attrs)
        status, error = "ok", None
        start_ts = datetime.now(UTC).isoformat()
        t0 = time.perf_counter()
        try:
            yield record
        except BaseException as e:
            status, error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            _current_span.reset(token)
            self._emit(
                {
                    "trace_id": self.trace_id,
                    "span_id": span_id,
                    "parent_id": parent_id,
                    "name": name,
                    "start_ts": start_ts,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                    "status": status,
                    **({"error": error} if error else {}),
                    **record,
                }
            )

    def _emit(self, span: dict[str, Any]) -> None:
        self.spans.append(span)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(span, ensure_ascii=False, default=str) + "\n")


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load all spans from a JSONL trace file.

    Args:
        path: Trace file.

    Returns:
        Spans in the order they finished.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TracedClient:
    """Backend wrapper that records a span for every model call.

    Each span's ``key`` is the cassette key of the call (the same digests as the replay
    client), so a trace row points at the exact cassette file that reproduces it.

    Args:
        backend: Wrapped backend (typically a :class:`~paper2flow.llm.replay.ReplayClient`).
        tracer: Destination tracer.
        digests: Pinned model digests (``models.lock``) used in cassette keys.
    """

    def __init__(
        self, backend: Backend, tracer: Tracer, digests: Mapping[str, str] | None = None
    ) -> None:
        self.backend = backend
        self.tracer = tracer
        self.digests = dict(digests or {})

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run and trace a chat request.

        Args:
            request: The chat request.

        Returns:
            The wrapped backend's response.
        """
        key = request.cache_key(self.digests.get(request.model))
        with self.tracer.span("llm.chat", model=request.model, key=key) as s:
            response = self.backend.chat(request)
            s.update(
                cached=response.cached,
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
                load_ms=response.usage.load_ms,
                n_tool_calls=len(response.tool_calls),
            )
            return response

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        """Run and trace an image request.

        Args:
            request: The image request.

        Returns:
            The wrapped backend's response.
        """
        key = request.cache_key(self.digests.get(request.model))
        with self.tracer.span("llm.image", model=request.model, key=key) as s:
            response = self.backend.generate_image(request)
            s.update(cached=response.cached, image_sha256=response.sha256())
            return response
