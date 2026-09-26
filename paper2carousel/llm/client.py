"""Thin client for Ollama's native HTTP API, plus the backend protocol the pipeline uses."""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import httpx

from paper2carousel.llm.types import ChatRequest, ChatResponse, ImageRequest, ImageResponse


class OllamaError(RuntimeError):
    """Raised when Ollama returns an error or cannot be reached."""


@runtime_checkable
class Backend(Protocol):
    """Anything that can serve chat and image requests.

    Implemented by :class:`OllamaClient` and by the wrappers in
    :mod:`paper2carousel.llm.replay` and :mod:`paper2carousel.tracing`, so they compose.
    """

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run a chat request."""
        ...

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        """Run a text-to-image request."""
        ...


def normalize_tag(model: str) -> str:
    """Add the implicit ``:latest`` tag, as Ollama does.

    Args:
        model: Model name with or without a tag.

    Returns:
        The fully qualified tag.
    """
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


class OllamaClient:
    """Synchronous client for a local Ollama server.

    Args:
        host: Base URL of the Ollama server.
        timeout_s: Per-request timeout. Cold-loading a large model can take a while.
        transport: Optional httpx transport, used by tests to mock the server.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout_s: float = 900.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(base_url=host, timeout=timeout_s, transport=transport)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> OllamaClient:
        """Enter a context that closes the client on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client."""
        self.close()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            response = self._http.request(method, path, json=payload)
        except httpx.HTTPError as e:
            raise OllamaError(f"Cannot reach Ollama at {self._http.base_url}: {e}") from e
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            raise OllamaError(f"{method} {path} -> {response.status_code}: {detail}")
        return response.json()

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run a chat request via ``POST /api/chat``.

        Args:
            request: The chat request.

        Returns:
            The model's response.
        """
        return ChatResponse.from_ollama(self._request("POST", "/api/chat", request.to_payload()))

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        """Generate an image via ``POST /api/generate`` (experimental in Ollama).

        Args:
            request: The image request.

        Returns:
            The generated image.
        """
        data = self._request("POST", "/api/generate", request.to_payload())
        return ImageResponse.from_ollama(data)

    def unload(self, model: str, wait_s: float = 0.0, poll_s: float = 0.5) -> bool:
        """Ask Ollama to unload a model (``keep_alive=0``), optionally waiting until it's gone.

        Unloading is asynchronous: the request returns before memory is freed, so callers
        that are about to load another large model should wait.

        Args:
            model: Model tag.
            wait_s: Maximum time to wait for the model to leave ``/api/ps`` (0 = don't wait).
            poll_s: Polling interval while waiting.

        Returns:
            True if the model is confirmed gone (always True when not waiting).
        """
        self._request("POST", "/api/generate", {"model": model, "keep_alive": 0})
        if wait_s <= 0:
            return True
        deadline = time.monotonic() + wait_s
        while model in self.running_models():
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_s)
        return True

    def list_models(self) -> dict[str, str]:
        """List locally installed models.

        Returns:
            Mapping of model tag to digest.
        """
        data = self._request("GET", "/api/tags")
        return {m["name"]: m["digest"] for m in data.get("models", [])}

    def running_models(self) -> list[str]:
        """List models currently loaded in memory.

        Returns:
            Model tags.
        """
        data = self._request("GET", "/api/ps")
        return [m["name"] for m in data.get("models", [])]

    def show(self, model: str) -> dict[str, Any]:
        """Return model metadata, including declared ``capabilities`` on recent Ollama.

        Args:
            model: Model tag.

        Returns:
            The raw ``/api/show`` response.
        """
        result: dict[str, Any] = self._request("POST", "/api/show", {"model": model})
        return result

    def version(self) -> str:
        """Return the Ollama server version.

        Returns:
            Version string, e.g. ``"0.24.0"``.
        """
        return str(self._request("GET", "/api/version").get("version", "unknown"))
