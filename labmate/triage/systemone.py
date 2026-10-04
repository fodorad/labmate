"""The decision-model endpoint of Ollama (``/v1/systemone``).

A decision model does not write text. It gets a ``state`` (the text to judge) and typed
``questions`` and scores every option of every question in one pass, which makes it fast and its
answers probabilities: a ``choice`` among named options, a ``score`` on ordered levels, or a
``noul`` (true or false) question. Ollama refuses such a model on ``/api/chat``.
"""

from __future__ import annotations

from typing import Any

import httpx

from labmate.config import Config

DECISION = "decision"
"""The capability Ollama lists for a decision model (``/api/show``)."""

TIMEOUT_S = 120.0
"""Seconds to wait for one decision (loading the model included)."""


class DecisionError(RuntimeError):
    """Ollama refused the request or answered without the expected answers."""


def has_decision_capability(
    config: Config, model: str, transport: httpx.BaseTransport | None = None
) -> bool:
    """Whether Ollama lists the model as a decision model.

    Args:
        config: Loaded configuration (the Ollama host).
        model: Model tag.
        transport: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        ``True`` if its capabilities include ``decision``.
    """
    with httpx.Client(transport=transport, timeout=TIMEOUT_S) as http:
        response = http.post(f"{config.ollama.host}/api/show", json={"model": model})
    if response.status_code == httpx.codes.NOT_FOUND:
        raise DecisionError(f"model {model!r} is not installed (ollama pull {model})")
    response.raise_for_status()
    return DECISION in response.json().get("capabilities", [])


def systemone(
    config: Config,
    model: str,
    state: str,
    questions: dict[str, Any],
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Ask a decision model typed questions about a text.

    Args:
        config: Loaded configuration (the Ollama host).
        model: A decision model tag, e.g. ``clef-flash``.
        state: The text to judge.
        questions: Question name to its definition (``type``, ``instructions``, ``criteria``).
        transport: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        The ``answers`` object: question name to ``choice``, ``score`` or ``probability`` with
        the probability of every option.

    Raises:
        DecisionError: If Ollama returns an error or no answers.
    """
    with httpx.Client(transport=transport, timeout=TIMEOUT_S) as http:
        response = http.post(
            f"{config.ollama.host}/v1/systemone",
            json={"model": model, "state": state, "questions": questions},
        )
    if response.status_code != httpx.codes.OK:
        raise DecisionError(f"{model}: {response.status_code} {response.text[:200]}")
    answers = response.json().get("answers")
    if not answers:
        raise DecisionError(f"{model} returned no answers")
    return dict(answers)
