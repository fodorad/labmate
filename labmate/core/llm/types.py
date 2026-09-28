"""Request/response types for model calls, independent of any HTTP client.

The request types know how to serialise themselves to Ollama's JSON payloads and how to
derive a stable cache key, which the replay layer uses to record and replay calls.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

NS_PER_MS = 1_000_000
"""Ollama reports durations in nanoseconds."""


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to JSON with sorted keys and no whitespace.

    Args:
        obj: Any JSON-serialisable object.

    Returns:
        A deterministic JSON string, so equal objects always produce equal strings.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _cache_key(kind: str, payload: dict[str, Any], digest: str | None) -> str:
    material = {"kind": kind, "digest": digest or payload["model"], "payload": payload}
    return hashlib.sha256(canonical_json(material).encode()).hexdigest()


def _sorted(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sorted(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted(v) for v in value]
    return value


class ToolFunction(BaseModel):
    """A function invocation requested by the model.

    Arguments are stored with sorted keys. Cassettes are written with sorted keys, so
    without this a live run and its replay would see the same call in different key
    orders, and requests that echo the call back to the model would get different keys.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments")
    @classmethod
    def _canonical(cls, value: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = _sorted(value)
        return result


class ToolCall(BaseModel):
    """A single tool call emitted by the model."""

    function: ToolFunction


class Message(BaseModel):
    """One chat message.

    Attributes:
        role: Speaker role.
        content: Text content.
        tool_calls: Tool calls made by the assistant in this message.
        tool_name: For ``role="tool"``: the tool whose result this message carries.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = None


class ChatRequest(BaseModel):
    """A chat completion request.

    Attributes:
        model: Ollama model tag.
        messages: Conversation so far.
        format: ``"json"`` or a JSON schema for constrained decoding.
        tools: Tool definitions in Ollama/OpenAI function format.
        temperature: Sampling temperature.
        seed: Sampling seed.
        num_ctx: Context window override.
        num_predict: Maximum number of tokens to generate (``None`` = model default).
        think: Enable or disable the model's thinking mode (``None`` = model default).
        keep_alive: How long the model stays loaded after the call. Not part of the cache
            key, because it does not change the output.
    """

    model: str
    messages: list[Message]
    format: dict[str, Any] | Literal["json"] | None = None
    tools: list[dict[str, Any]] | None = None
    temperature: float = 0.0
    seed: int | None = 42
    num_ctx: int | None = None
    num_predict: int | None = None
    think: bool | None = False
    keep_alive: str | int | None = None

    def to_payload(self) -> dict[str, Any]:
        """Build the JSON body for ``POST /api/chat``.

        Returns:
            The request payload (non-streaming).
        """
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "stream": False,
            "options": options,
        }
        if self.format is not None:
            payload["format"] = self.format
        if self.tools is not None:
            payload["tools"] = self.tools
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def cache_key(self, digest: str | None = None) -> str:
        """Stable key identifying this request's output.

        Args:
            digest: Pinned model digest. When given, it replaces the model tag in the key,
                so re-tagging a model does not invalidate cassettes but a new model version
                does.

        Returns:
            A sha256 hex digest.
        """
        payload = self.to_payload()
        payload.pop("keep_alive", None)
        return _cache_key("chat", payload, digest)


class Usage(BaseModel):
    """Token counts and timings for one call (milliseconds)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    load_ms: float = 0.0
    prompt_ms: float = 0.0
    eval_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def tokens_per_s(self) -> float:
        """Generation throughput, or 0.0 when no eval time was reported."""
        if self.eval_ms <= 0:
            return 0.0
        return self.completion_tokens / (self.eval_ms / 1000)

    @classmethod
    def from_ollama(cls, data: dict[str, Any]) -> Usage:
        """Extract usage from an Ollama response body.

        Args:
            data: Raw response JSON.

        Returns:
            The parsed usage; missing fields default to zero.
        """
        return cls(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            load_ms=data.get("load_duration", 0) / NS_PER_MS,
            prompt_ms=data.get("prompt_eval_duration", 0) / NS_PER_MS,
            eval_ms=data.get("eval_duration", 0) / NS_PER_MS,
            total_ms=data.get("total_duration", 0) / NS_PER_MS,
        )


class ChatResponse(BaseModel):
    """A chat completion result.

    Attributes:
        model: Model that produced the response.
        content: Assistant text.
        thinking: Thinking trace, if the model produced one.
        tool_calls: Tool calls requested by the model.
        usage: Token counts and timings.
        cached: True if served from the cassette store instead of the model.
    """

    model: str
    content: str = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cached: bool = False

    @classmethod
    def from_ollama(cls, data: dict[str, Any]) -> ChatResponse:
        """Parse a ``/api/chat`` response body.

        Args:
            data: Raw response JSON.

        Returns:
            The parsed response.
        """
        message = data.get("message", {})
        return cls(
            model=data.get("model", ""),
            content=message.get("content", ""),
            thinking=message.get("thinking") or None,
            tool_calls=[ToolCall.model_validate(tc) for tc in message.get("tool_calls") or []],
            usage=Usage.from_ollama(data),
        )


class EmbedRequest(BaseModel):
    """An embedding request (``POST /api/embed``): one vector per input text.

    Attributes:
        model: Embedding model tag.
        input: Texts to embed, in order.
        truncate: Let Ollama truncate inputs longer than the model's context.
        keep_alive: How long the model stays loaded (not part of the cache key).
    """

    model: str
    input: list[str]
    truncate: bool = True
    keep_alive: str | int | None = None

    def to_payload(self) -> dict[str, Any]:
        """Build the JSON body for ``POST /api/embed``.

        Returns:
            The request payload.
        """
        payload: dict[str, Any] = {"model": self.model, "input": self.input}
        payload["truncate"] = self.truncate
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def cache_key(self, digest: str | None = None) -> str:
        """Stable key identifying this request's output (see :meth:`ChatRequest.cache_key`).

        Args:
            digest: Pinned model digest.

        Returns:
            A sha256 hex digest.
        """
        payload = self.to_payload()
        payload.pop("keep_alive", None)
        return _cache_key("embed", payload, digest)


class EmbedResponse(BaseModel):
    """Embeddings for an :class:`EmbedRequest`.

    Attributes:
        model: Model that produced them.
        embeddings: One vector per input text, in input order.
        prompt_tokens: Tokens read.
        total_ms: Wall-clock time reported by Ollama.
        cached: True if served from the cassette store.
    """

    model: str
    embeddings: list[list[float]]
    prompt_tokens: int = 0
    total_ms: float = 0.0
    cached: bool = False

    @classmethod
    def from_ollama(cls, data: dict[str, Any]) -> EmbedResponse:
        """Parse a ``/api/embed`` response body.

        Args:
            data: Raw response JSON.

        Returns:
            The parsed response.
        """
        return cls(
            model=data.get("model", ""),
            embeddings=data.get("embeddings", []),
            prompt_tokens=data.get("prompt_eval_count", 0),
            total_ms=data.get("total_duration", 0) / NS_PER_MS,
        )
