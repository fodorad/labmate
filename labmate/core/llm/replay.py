"""Record/replay cache for model calls and web requests ("cassettes").

Every model request is keyed by :meth:`ChatRequest.cache_key` (sha256 over the model digest
and the canonical payload); every web request (paper downloads, search APIs) by its method,
URL and body. In ``replay`` mode neither a model nor the network is needed, which is what
makes runs reproducible byte for byte and lets the tests run every feature offline.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from labmate.config import ReplayMode
from labmate.core.llm.client import Backend
from labmate.core.llm.types import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    canonical_json,
)

log = logging.getLogger(__name__)


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
            The stored response dict, or ``None`` if absent or unreadable (a cassette cut
            short by an interrupted run counts as missing, so it is recorded again).
        """
        p = self.path(key)
        if not p.exists():
            return None
        try:
            response: dict[str, Any] = json.loads(p.read_text())["response"]
        except (json.JSONDecodeError, KeyError):
            log.warning("ignoring unreadable cassette %s", p)
            return None
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
        text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        # write a temporary file and rename it: an interrupted run never leaves half a cassette
        with tempfile.NamedTemporaryFile(
            "w", dir=p.parent, prefix=f".{key}.", suffix=".tmp", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text)
        Path(tmp.name).replace(p)


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

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Serve an embedding request from the cassette store or the live backend.

        Args:
            request: The embedding request.

        Returns:
            The response; ``cached`` is True when it came from a cassette.
        """
        key = request.cache_key(self.digests.get(request.model))
        hit = self._lookup(key)
        if hit is not None:
            return EmbedResponse.model_validate({**hit, "cached": True})
        response = self._live().embed(request)
        if self.mode is not ReplayMode.LIVE:
            self.store.put(key, request.to_payload(), response.model_dump(exclude={"cached"}))
        return response


_KEPT_HEADERS = ("content-type", "location")
"""Response headers stored with a recorded web response (enough to replay it)."""


def http_key(request: httpx.Request) -> str:
    """Cache key of a web request: sha256 over its method, URL and body.

    Args:
        request: The request.

    Returns:
        A sha256 hex digest.
    """
    material = {
        "kind": "http",
        "method": request.method,
        "url": str(request.url),
        "body": base64.b64encode(request.read()).decode(),
    }
    return hashlib.sha256(canonical_json(material).encode()).hexdigest()


class RecordedTransport(httpx.BaseTransport):
    """An ``httpx`` transport that records and replays web requests like model calls.

    Responses below 400 (including redirects) are stored; errors are not, so a retry after
    a temporary failure (arXiv answers 429 now and then) goes to the network again.

    Args:
        inner: The live transport. May be ``None`` in ``replay`` mode.
        store: Cassette store (shared with the model calls).
        mode: Replay mode.

    Raises:
        ValueError: If a live transport is required by ``mode`` but not given.
    """

    def __init__(
        self, inner: httpx.BaseTransport | None, store: CassetteStore, mode: ReplayMode
    ) -> None:
        if inner is None and mode is not ReplayMode.REPLAY:
            raise ValueError(f"mode={mode.value!r} needs a live transport")
        self.inner = inner
        self.store = store
        self.mode = mode

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Serve a request from the cassette store or the network.

        Args:
            request: The request.

        Returns:
            The response.

        Raises:
            CassetteMissError: In ``replay`` mode, if the request was never recorded.
        """
        key = http_key(request)
        if self.mode in (ReplayMode.AUTO, ReplayMode.REPLAY):
            hit = self.store.get(key)
            if hit is not None:
                return httpx.Response(
                    hit["status"],
                    headers=hit["headers"],
                    content=base64.b64decode(hit["body"]),
                    request=request,
                )
            if self.mode is ReplayMode.REPLAY:
                raise CassetteMissError(f"No cassette for {request.method} {request.url}")
        if self.inner is None:  # pragma: no cover - prevented by __init__
            raise RuntimeError("no live transport configured")
        response = self.inner.handle_request(request)
        body = response.read()
        if self.mode is not ReplayMode.LIVE and response.status_code < 400:
            headers = {k: v for k, v in response.headers.items() if k in _KEPT_HEADERS}
            self.store.put(
                key,
                {"method": request.method, "url": str(request.url)},
                {"status": response.status_code, "headers": headers,
                 "body": base64.b64encode(body).decode()},
            )  # fmt: skip
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
