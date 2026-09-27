"""Parsing model output into Pydantic models, with a lenient fallback.

Schema-constrained decoding (Ollama's ``format=``) is not honoured by every backend: the
capability probe showed the MLX builds return non-conforming text. The pipeline therefore
never trusts raw output. It tries a strict parse first, then a lenient one that strips
Markdown code fences and surrounding prose, and validates either way.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ValidationError

from paper2flow.llm.client import Backend
from paper2flow.llm.types import ChatRequest, Message

ParseMode = Literal["strict", "lenient"]
"""How a successful parse was achieved."""

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str | None:
    """Pull the most likely JSON object out of free-form model output.

    Order of attempts: the first fenced code block, then the span from the first ``{`` to
    the last ``}``.

    Args:
        text: Raw model output.

    Returns:
        The candidate JSON string, or ``None`` if nothing object-like is present.
    """
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_structured[M: BaseModel](
    text: str, model: type[M]
) -> tuple[M, ParseMode] | tuple[None, None]:
    """Validate model output against a schema, strictly first, then leniently.

    Args:
        text: Raw model output.
        model: Target Pydantic model class.

    Returns:
        ``(instance, mode)`` on success, ``(None, None)`` if neither attempt validates.
    """
    try:
        return model.model_validate_json(text), "strict"
    except ValidationError:
        pass
    candidate = extract_json(text)
    if candidate is not None:
        try:
            return model.model_validate_json(candidate), "lenient"
        except ValidationError:
            pass
    return None, None


def schema_instruction(model: type[BaseModel]) -> str:
    """System-prompt text that asks for JSON matching ``model``'s schema.

    Backends that ignore ``format=`` (the MLX builds) still follow an explicit instruction,
    so the pipeline sends both: the constraint where it is honoured, the instruction where
    it is not.

    Args:
        model: Target Pydantic model class.

    Returns:
        Instruction text including the JSON schema.
    """
    schema = json.dumps(model.model_json_schema(), separators=(",", ":"))
    return (
        "Respond with a single JSON object that validates against this JSON schema. "
        "Output only the JSON: no prose, no Markdown, no code fences.\n"
        f"Schema: {schema}"
    )


class StructuredOutputError(RuntimeError):
    """Raised when a model fails to produce schema-valid JSON within the retry budget."""


def _validation_error(text: str, model: type[BaseModel]) -> str:
    candidate = extract_json(text) or text
    try:
        model.model_validate_json(candidate)
    except ValidationError as e:
        return str(e)[:800]
    return "unknown validation error"  # pragma: no cover - only called after a failed parse


def structured_chat[M: BaseModel](
    backend: Backend,
    request: ChatRequest,
    model: type[M],
    max_retries: int = 2,
    check: Callable[[M], list[str]] | None = None,
) -> M:
    """Run a chat request and return its output validated as ``model``.

    Sends the schema both ways: ``format=`` (honoured by some backends) and a system
    instruction (followed by the rest). On invalid output it retries, showing the model its
    previous reply and the validation error, which is the cheapest self-correction loop.
    ``check`` adds semantic rules on top of the schema (a programmatic *gate*): any problems
    it returns are fed back exactly like schema errors.

    Args:
        backend: Any backend (usually traced + replayed).
        request: The base request; its ``format`` is overwritten with the schema.
        model: Target Pydantic model class.
        max_retries: Extra attempts after the first.
        check: Optional semantic validator returning a list of problems (empty = OK).

    Returns:
        The validated instance.

    Raises:
        StructuredOutputError: If no attempt validates.
    """
    messages = [Message(role="system", content=schema_instruction(model)), *request.messages]
    last = ""
    for _ in range(max_retries + 1):
        response = backend.chat(
            request.model_copy(update={"messages": messages, "format": model.model_json_schema()})
        )
        parsed, _ = parse_structured(response.content, model)
        last = response.content
        if parsed is not None:
            problems = check(parsed) if check else []
            if not problems:
                return parsed
            feedback = "That reply is valid JSON but breaks these rules:\n- " + "\n- ".join(
                problems
            )
        else:
            feedback = (
                f"That reply does not validate against the schema:\n"
                f"{_validation_error(last, model)}"
            )
        messages = [
            *messages,
            Message(role="assistant", content=last),
            Message(role="user", content=f"{feedback}\nReply again with only the corrected JSON."),
        ]
    raise StructuredOutputError(
        f"{request.model}: no valid {model.__name__} after {max_retries + 1} attempts; "
        f"last output: {last[:200]!r}"
    )
