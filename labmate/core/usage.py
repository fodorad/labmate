"""What the local models cost: time, tokens, loading and memory, call by call.

``ChatOllama`` returns Ollama's own timings with every reply (``response_metadata``). The
:class:`UsageCollector` is a LangChain callback that keeps them, so a benchmark can say how a run's
time was split between loading models, reading prompts and writing, and whether the models fit in
memory together. Set ``Config.callbacks`` and every model built by
:func:`labmate.core.chat.chat_model` reports to it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

LOAD_THRESHOLD_S = 0.5
"""A call that spent at least this long loading its model counts as a load."""

_NS = 1e-9


@dataclass(frozen=True)
class Call:
    """One model call, as Ollama timed it.

    Attributes:
        model: Model tag.
        wall_s: Seconds from the request to the reply.
        load_s: Seconds spent loading the model into memory.
        prompt_tokens: Tokens read.
        prompt_s: Seconds spent reading them.
        completion_tokens: Tokens written.
        decode_s: Seconds spent writing them.
    """

    model: str
    wall_s: float
    load_s: float
    prompt_tokens: int
    prompt_s: float
    completion_tokens: int
    decode_s: float


@dataclass(frozen=True)
class Usage:
    """The sum of a run's model calls.

    Attributes:
        calls: Number of model calls.
        models: Distinct models used.
        tokens_in: Tokens read.
        tokens_out: Tokens written.
        model_s: Seconds the models were busy (calls that ran in parallel add up).
        load_s: Seconds spent loading models.
        loads: Calls that had to load their model.
        swaps: Loads beyond one per model: a model that had to be loaded again.
        peak_gb: Most memory the loaded models took at once (0 if not measured).
        prompt_tps: Prompt tokens read per second.
        decode_tps: Tokens written per second.
    """

    calls: int
    models: tuple[str, ...]
    tokens_in: int
    tokens_out: int
    model_s: float
    load_s: float
    loads: int
    swaps: int
    peak_gb: float
    prompt_tps: float
    decode_tps: float


def ollama_resident(host: str, transport: httpx.BaseTransport | None = None) -> Callable[[], float]:
    """A function that returns the gigabytes of models Ollama has loaded right now.

    Args:
        host: The Ollama server address.
        transport: An httpx transport to reach it through (tests).

    Returns:
        ``() -> GB``; 0.0 if the server cannot be asked.
    """

    def resident() -> float:
        try:
            with httpx.Client(transport=transport, timeout=5) as http:
                loaded = http.get(f"{host}/api/ps").json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return 0.0
        return sum(m.get("size", 0) for m in loaded) / 1e9

    return resident


class UsageCollector(BaseCallbackHandler):
    """Records every chat-model call of a run.

    Args:
        resident: Returns the gigabytes of models loaded now (see :func:`ollama_resident`);
            polled after each call to find the peak.
    """

    def __init__(self, resident: Callable[[], float] | None = None) -> None:
        self._resident = resident
        self._lock = threading.Lock()
        self._started: dict[UUID, float] = {}
        self._calls: list[Call] = []
        self._peak_gb = 0.0

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, **_: Any
    ) -> None:
        """Note when a call starts."""
        with self._lock:
            self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **_: Any) -> None:
        """Keep the timings Ollama returned with the reply."""
        generation = response.generations[0][0]
        meta = getattr(getattr(generation, "message", None), "response_metadata", {}) or {}
        gb = self._resident() if self._resident else 0.0
        with self._lock:
            started = self._started.pop(run_id, time.perf_counter())
            self._peak_gb = max(self._peak_gb, gb)
            self._calls.append(
                Call(
                    model=str(meta.get("model", "")),
                    wall_s=time.perf_counter() - started,
                    load_s=meta.get("load_duration", 0) * _NS,
                    prompt_tokens=meta.get("prompt_eval_count", 0),
                    prompt_s=meta.get("prompt_eval_duration", 0) * _NS,
                    completion_tokens=meta.get("eval_count", 0),
                    decode_s=meta.get("eval_duration", 0) * _NS,
                )
            )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        """Forget a call that failed."""
        with self._lock:
            self._started.pop(run_id, None)

    def summary(self) -> Usage:
        """The sum of the calls so far.

        Returns:
            The totals.
        """
        with self._lock:
            calls, peak = list(self._calls), self._peak_gb
        loads = sum(c.load_s >= LOAD_THRESHOLD_S for c in calls)
        models = tuple(dict.fromkeys(c.model for c in calls))
        prompt_s, decode_s = sum(c.prompt_s for c in calls), sum(c.decode_s for c in calls)
        return Usage(
            calls=len(calls),
            models=models,
            tokens_in=sum(c.prompt_tokens for c in calls),
            tokens_out=sum(c.completion_tokens for c in calls),
            model_s=sum(c.wall_s for c in calls),
            load_s=sum(c.load_s for c in calls),
            loads=loads,
            swaps=max(0, loads - len(models)),
            peak_gb=peak,
            prompt_tps=sum(c.prompt_tokens for c in calls) / prompt_s if prompt_s else 0.0,
            decode_tps=sum(c.completion_tokens for c in calls) / decode_s if decode_s else 0.0,
        )
