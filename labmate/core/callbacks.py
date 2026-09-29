"""One LangChain callback handler writes labmate's trace for chains and graphs.

Every Runnable (a chain, a step, a graph node, a chat model call) reports its start and end
to the callback handlers in its config, with its own run id and its parent's. LangSmith's
tracer listens to the same events when ``LANGSMITH_TRACING=true``. This handler keeps the
part of the run tree worth reading and writes it to ``trace.jsonl``:

- the root run, as a ``run`` span;
- the pipeline steps (Runnables named with :func:`~labmate.core.lc.step`), as ``step.<name>``;
- every chat model call, as an ``llm.chat`` span with its model, tokens, cassette key and
  whether it was replayed.

Internal runs (prompt templates, sequences, parsers) are skipped; their children are
attached to the nearest recorded ancestor. A step adds summary attributes (e.g. how many
claims it verified) with the custom event :data:`ATTRS_EVENT`.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult

from labmate.core.lc import ATTRS_EVENT, STEP_KEY
from labmate.core.tracing import Tracer


@dataclass
class _Open:
    """A recorded run that has started but not finished."""

    span_id: str
    parent_id: str | None
    name: str
    start_ts: str
    t0: float
    attrs: dict[str, Any] = field(default_factory=dict)


class TraceHandler(BaseCallbackHandler):
    """Writes the root run, the named steps and the model calls of a run as spans.

    Args:
        tracer: Destination (its ``trace_id`` is shared by all spans).
        attrs: Attributes for the root ``run`` span, e.g. ``{"paper": "1706.03762"}``.
    """

    def __init__(self, tracer: Tracer, attrs: dict[str, Any] | None = None) -> None:
        self.tracer = tracer
        self.attrs = dict(attrs or {})
        self._open: dict[UUID, _Open] = {}
        self._skipped: dict[UUID, str | None] = {}  # run -> nearest recorded ancestor span
        self._lock = threading.Lock()  # batched runs report from worker threads

    # --- bookkeeping --------------------------------------------------------------------

    def _span_of(self, run_id: UUID | None) -> str | None:
        if run_id is None:
            return None
        if run_id in self._open:
            return self._open[run_id].span_id
        return self._skipped.get(run_id)

    def _start(self, run_id: UUID, parent: UUID | None, name: str, attrs: dict[str, Any]) -> None:
        with self._lock:
            self._open[run_id] = _Open(
                span_id=uuid.uuid4().hex[:16],
                parent_id=self._span_of(parent),
                name=name,
                start_ts=datetime.now(UTC).isoformat(),
                t0=time.perf_counter(),
                attrs=attrs,
            )

    def _skip(self, run_id: UUID, parent: UUID | None) -> None:
        with self._lock:
            self._skipped[run_id] = self._span_of(parent)

    def _end(self, run_id: UUID, error: BaseException | None = None, **attrs: Any) -> None:
        with self._lock:
            self._skipped.pop(run_id, None)
            run = self._open.pop(run_id, None)
        if run is None:
            return
        span = {
            "trace_id": self.tracer.trace_id,
            "span_id": run.span_id,
            "parent_id": run.parent_id,
            "name": run.name,
            "start_ts": run.start_ts,
            "latency_ms": round((time.perf_counter() - run.t0) * 1000, 3),
            "status": "error" if error else "ok",
            **({"error": f"{type(error).__name__}: {error}"} if error else {}),
            **run.attrs,
            **attrs,
        }
        self.tracer.record(span)

    # --- chains, steps and graph nodes --------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Open a ``run`` span for the root, a ``step.<name>`` span for a named step."""
        name = kwargs.get("name") or (serialized or {}).get("name", "")
        if parent_run_id is None:
            self._start(run_id, None, "run", {"command": name, **self.attrs})
        elif (metadata or {}).get(STEP_KEY) == name:
            self._start(run_id, parent_run_id, f"step.{name}", {})
        else:
            self._skip(run_id, parent_run_id)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Close the run's span, if it has one."""
        self._end(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Close the run's span with the error."""
        self._end(run_id, error)

    def on_custom_event(self, name: str, data: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Attach a step's summary attributes (:data:`ATTRS_EVENT`) to its span."""
        if name != ATTRS_EVENT:
            return
        with self._lock:
            if run_id in self._open:
                self._open[run_id].attrs.update(data)

    # --- model calls --------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Open an ``llm.chat`` span."""
        self._start(run_id, parent_run_id, "llm.chat", {})

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Close the ``llm.chat`` span with the model, tokens, cassette key and cache hit."""
        generation = response.generations[0][0]
        attrs: dict[str, Any] = {}
        if isinstance(generation, ChatGeneration):
            meta = generation.message.response_metadata
            usage = getattr(generation.message, "usage_metadata", None) or {}
            attrs = {
                "model": meta.get("model"),
                "key": meta.get("cache_key"),
                "cached": meta.get("cached", False),
                "tokens_in": usage.get("input_tokens", 0),
                "tokens_out": usage.get("output_tokens", 0),
            }
        self._end(run_id, **attrs)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Close the ``llm.chat`` span with the error."""
        self._end(run_id, error)
