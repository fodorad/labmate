"""The Ollama server: every call that is not a chat or an embedding.

Chat traffic goes through :mod:`labmate.core.chat`. This module asks the server what is installed
and loaded, unloads models, and serves decision models.

A decision model does not write text. It gets a ``state`` (the text to judge) and typed
``questions`` and scores every option of every question in one pass, which makes it fast and its
answers probabilities: a ``choice`` among named options, a ``score`` on ordered levels, or a
``noul`` (true or false) question. Ollama refuses such a model on ``/api/chat`` and serves it on
``/v1/systemone``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from labmate.config import Config

TIMEOUT_S = 120.0
"""Seconds to wait for one request (loading a model included)."""

QUICK_TIMEOUT_S = 5.0
"""Seconds to wait for a status question (what is loaded, which version)."""


class DecisionError(RuntimeError):
    """Ollama refused a decision request, or the model is not installed."""


@dataclass(frozen=True)
class OllamaServer:
    """One Ollama server.

    Attributes:
        host: The server address, e.g. ``http://localhost:11434``.
        transport: An httpx transport to reach it through (tests); the network if ``None``.
    """

    host: str
    transport: httpx.BaseTransport | None = None

    @classmethod
    def from_config(
        cls, config: Config, transport: httpx.BaseTransport | None = None
    ) -> OllamaServer:
        """The configured server (``[ollama].host`` or ``LABMATE_OLLAMA_HOST``).

        Args:
            config: Loaded configuration.
            transport: An httpx transport to reach it through (tests).

        Returns:
            The server.
        """
        return cls(config.ollama.host, transport)

    def _client(self, timeout: float = TIMEOUT_S) -> httpx.Client:
        return httpx.Client(base_url=self.host, transport=self.transport, timeout=timeout)

    def loaded(self) -> dict[str, float]:
        """The models loaded now.

        Returns:
            Model name to gigabytes.
        """
        with self._client(QUICK_TIMEOUT_S) as http:
            models = http.get("/api/ps").json().get("models", [])
        return {m["name"]: m.get("size", 0) / 1e9 for m in models}

    def resident_gb(self) -> float:
        """The gigabytes of models loaded now; 0.0 if the server cannot be asked.

        Returns:
            Gigabytes.
        """
        try:
            return round(sum(self.loaded().values()), 3)
        except (httpx.HTTPError, ValueError):
            return 0.0

    def unload_all(self) -> None:
        """Unload every loaded model, so the next call is a cold start."""
        with self._client() as http:
            for loaded in http.get("/api/ps").json().get("models", []):
                http.post("/api/generate", json={"model": loaded["name"], "keep_alive": 0})

    def sizes(self) -> dict[str, float]:
        """The size of every installed model.

        Returns:
            Model tag to gigabytes; a ``name:latest`` model is also listed as ``name``.
        """
        with self._client() as http:
            models = http.get("/api/tags").json().get("models", [])
        sizes = {m["name"]: m["size"] / 1e9 for m in models}
        sizes.update({name.removesuffix(":latest"): gb for name, gb in sizes.items()})
        return sizes

    def capabilities(self, model: str) -> list[str]:
        """What a model can do, e.g. ``completion``, ``tools`` or ``decision``.

        Args:
            model: Model tag.

        Returns:
            The capabilities Ollama lists for it.

        Raises:
            DecisionError: If the model is not installed.
        """
        with self._client() as http:
            response = http.post("/api/show", json={"model": model})
        if response.status_code == httpx.codes.NOT_FOUND:
            raise DecisionError(f"model {model!r} is not installed (ollama pull {model})")
        response.raise_for_status()
        return list(response.json().get("capabilities", []))

    def version(self) -> str:
        """The server version.

        Returns:
            e.g. ``0.30.0``; ``unknown`` if the server cannot be asked.
        """
        try:
            with self._client(QUICK_TIMEOUT_S) as http:
                return str(http.get("/api/version").json().get("version", "unknown"))
        except (httpx.HTTPError, ValueError):
            return "unknown"

    def systemone(self, model: str, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        """Ask a decision model typed questions about a text.

        Args:
            model: A decision model tag, e.g. ``clef-flash``.
            state: The text to judge.
            questions: Question name to its definition (``type``, ``instructions``,
                ``criteria``).

        Returns:
            The ``answers`` object: question name to ``choice``, ``score`` or ``probability``
            with the probability of every option.

        Raises:
            DecisionError: If Ollama returns an error or no answers.
        """
        with self._client() as http:
            response = http.post(
                "/v1/systemone", json={"model": model, "state": state, "questions": questions}
            )
        if response.status_code != httpx.codes.OK:
            raise DecisionError(f"{model}: {response.status_code} {response.text[:200]}")
        answers = response.json().get("answers")
        if not answers:
            raise DecisionError(f"{model} returned no answers")
        return dict(answers)
