"""Parsing model output into Pydantic models, with a lenient fallback.

Schema-constrained decoding (Ollama's ``format``) is not honoured by every model build: the
MLX builds return non-conforming text. A reply is therefore never trusted as is. It is
parsed strictly first, then leniently (Markdown code fences and surrounding prose stripped),
and validated either way.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(RuntimeError):
    """Raised when a model fails to produce schema-valid JSON within the retry budget."""


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


def parse_structured[M: BaseModel](text: str, model: type[M]) -> M | None:
    """Validate model output against a schema, strictly first, then leniently.

    Args:
        text: Raw model output.
        model: Target Pydantic model class.

    Returns:
        The instance, or ``None`` if neither attempt validates.
    """
    for candidate in (text, extract_json(text)):
        if candidate is None:
            continue
        try:
            return model.model_validate_json(candidate)
        except ValidationError:
            continue
    return None


def schema_instruction(model: type[BaseModel]) -> str:
    """System-prompt text that asks for JSON matching ``model``'s schema.

    Model builds that ignore the ``format`` constraint still follow an explicit instruction,
    so both are sent: the constraint where it is honoured, the instruction where it is not.

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


def validation_error(text: str, model: type[BaseModel]) -> str:
    """The validation error of a reply, as shown to the model when it retries.

    Args:
        text: Raw model output that failed to validate.
        model: Target Pydantic model class.

    Returns:
        The error text (truncated).
    """
    candidate = extract_json(text) or text
    try:
        model.model_validate_json(candidate)
    except ValidationError as e:
        return str(e)[:800]
    return "unknown validation error"  # pragma: no cover - only called after a failed parse
