import contextvars
import threading

import pytest

from paper2flow.parallel import parallel_map

VAR: contextvars.ContextVar[str] = contextvars.ContextVar("VAR", default="unset")


def test_order_is_preserved_and_work_is_concurrent():
    seen = set()

    def f(x):
        seen.add(threading.get_ident())
        return x * 2

    assert parallel_map(f, list(range(20)), workers=4) == [x * 2 for x in range(20)]


def test_context_is_propagated_to_workers():
    VAR.set("parent")
    assert parallel_map(lambda _: VAR.get(), [1, 2, 3], workers=3) == ["parent"] * 3


def test_single_worker_runs_inline():
    assert (
        parallel_map(lambda x: threading.get_ident(), [1, 2], workers=1)
        == [threading.get_ident()] * 2
    )


def test_first_exception_propagates():
    def boom(x):
        raise KeyError(x)

    with pytest.raises(KeyError):
        parallel_map(boom, [1, 2], workers=2)
