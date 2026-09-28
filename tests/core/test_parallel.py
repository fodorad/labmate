import contextvars

import pytest

from labmate.core.parallel import parallel_map

VAR: contextvars.ContextVar[str] = contextvars.ContextVar("VAR", default="unset")


def test_results_keep_the_input_order():
    assert parallel_map(lambda x: x * 2, list(range(20)), workers=4) == [x * 2 for x in range(20)]


def test_context_is_propagated_to_workers():
    VAR.set("parent")
    assert parallel_map(lambda _: VAR.get(), [1, 2, 3], workers=3) == ["parent"] * 3


def test_first_exception_propagates():
    def boom(x):
        raise KeyError(x)

    with pytest.raises(KeyError):
        parallel_map(boom, [1, 2], workers=2)
