"""Record/replay cache for model calls ("cassettes").

Every request is keyed by :meth:`ChatRequest.cache_key` (sha256 over the model digest and
the canonical payload). In ``replay`` mode no model is needed at all, which is what makes
published runs reproducible byte-for-byte and lets CI run the full pipeline without Ollama.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from paper2carousel.config import ReplayMode
from paper2carousel.llm.client import Backend
from paper2carousel.llm.types import (
    ChatRequest,
    ChatResponse,
    ImageRequest,
    ImageResponse,
)


class CassetteMissError(LookupError):
    """Raised in ``replay`` mode when no cassette exists for a request."""


class CassetteStore:
    """Directory of cassettes, one JSON file per request, sharded by key prefix.

    Args:
        root: Directory holding the cassettes.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: str) -> Path:
        """Return the file path for a cache key.

        Args:
            key: sha256 hex cache key.

        Returns:
            ``<root>/<key[:2]>/<key>.json``.
        """
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Load a stored response.

        Args:
            key: Cache key.

        Returns:
            The stored response dict, or ``None`` if absent.
        """
        p = self.path(key)
        if not p.exists():
            return None
        response: dict[str, Any] = json.loads(p.read_text())["response"]
        return response

    def put(self, key: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        """Store a request/response pair. The request is kept for human inspection.

        Args:
            key: Cache key.
            request: Request payload.
            response: Response to store.
        """
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "request": request, "response": response}
        p.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


class ReplayClient:
    """Backend wrapper that records and replays calls according to a :class:`ReplayMode`.

    Args:
        backend: The live backend. May be ``None`` in ``replay`` mode.
        store: Cassette store.
        mode: Replay mode.
        digests: Pinned model digests (from ``models.lock``) used in cache keys.

    Raises:
        ValueError: If a live backend is required by ``mode`` but not given.
    """

    def __init__(
        self,
        backend: Backend | None,
        store: CassetteStore,
        mode: ReplayMode = ReplayMode.AUTO,
        digests: Mapping[str, str] | None = None,
    ) -> None:
        if backend is None and mode is not ReplayMode.REPLAY:
            raise ValueError(f"mode={mode.value!r} needs a live backend")
        self.backend = backend
        self.store = store
        self.mode = mode
        self.digests = dict(digests or {})

    def _live(self) -> Backend:
        if self.backend is None:  # pragma: no cover - prevented by __init__
            raise RuntimeError("no live backend configured")
        return self.backend

    def _lookup(self, key: str) -> dict[str, Any] | None:
        if self.mode in (ReplayMode.AUTO, ReplayMode.REPLAY):
            hit = self.store.get(key)
            if hit is not None:
                return hit
            if self.mode is ReplayMode.REPLAY:
                raise CassetteMissError(f"No cassette for key {key} in {self.store.root}")
        return None

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Serve a chat request from the cassette store or the live backend.

        Args:
            request: The chat request.

        Returns:
            The response; ``cached`` is True when it came from a cassette.
        """
        key = request.cache_key(self.digests.get(request.model))
        hit = self._lookup(key)
        if hit is not None:
            return ChatResponse.model_validate({**hit, "cached": True})
        response = self._live().chat(request)
        if self.mode is not ReplayMode.LIVE:
            self.store.put(key, request.to_payload(), response.model_dump(exclude={"cached"}))
        return response

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        """Serve an image request from the cassette store or the live backend.

        Args:
            request: The image request.

        Returns:
            The response; ``cached`` is True when it came from a cassette.
        """
        key = request.cache_key(self.digests.get(request.model))
        hit = self._lookup(key)
        if hit is not None:
            return ImageResponse.model_validate({**hit, "cached": True})
        response = self._live().generate_image(request)
        if self.mode is not ReplayMode.LIVE:
            self.store.put(key, request.to_payload(), response.model_dump(exclude={"cached"}))
        return response


def read_lock(path: Path) -> dict[str, str]:
    """Read pinned model digests.

    Args:
        path: Path to ``models.lock`` (JSON mapping of tag to digest).

    Returns:
        The mapping, or an empty dict if the file does not exist.
    """
    if not path.exists():
        return {}
    data: dict[str, str] = json.loads(path.read_text())
    return data


def write_lock(path: Path, digests: Mapping[str, str]) -> None:
    """Write pinned model digests.

    Args:
        path: Destination file.
        digests: Mapping of model tag to digest.
    """
    path.write_text(json.dumps(dict(digests), indent=2, sort_keys=True) + "\n")
