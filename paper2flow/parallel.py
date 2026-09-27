"""Bounded parallel map that keeps tracing context (the *parallelisation* pattern).

Worker threads don't inherit :mod:`contextvars` by default, so spans opened inside them
would lose their parent. Each task runs in a copy of the caller's context instead.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor


def parallel_map[T, R](fn: Callable[[T], R], items: Sequence[T], workers: int = 2) -> list[R]:
    """Apply ``fn`` to every item with at most ``workers`` in flight; keep input order.

    Args:
        fn: Function to apply.
        items: Inputs.
        workers: Maximum concurrency. Local models serve few requests at once
            (``OLLAMA_NUM_PARALLEL``), so this stays small.

    Returns:
        Results in the order of ``items``. The first exception raised by any task
        propagates.
    """
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, fn, item) for item in items]
        return [f.result() for f in futures]
