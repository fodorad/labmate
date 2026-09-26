"""Shared fixtures: an in-memory fake of the Ollama HTTP API."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pymupdf
import pytest

from paper2carousel.llm.client import OllamaClient
from paper2carousel.probe import ProbeClaim, solid_png
from paper2carousel.schemas import Deck, DraftSlide

INSTALLED = {
    "qwen3.6:35b-mlx": "1b50c6fdc2d4" + "0" * 52,
    "gemma4:26b-mlx": "21c59a2eae30" + "0" * 52,
    "no-tools:latest": "f0ad3edce8e4" + "0" * 52,
    "gemma4:e4b": "e4be4be4be4b" + "0" * 52,
    "x/z-image-turbo:latest": "77b78ce4e883" + "0" * 52,
    "x/flux2-klein:latest": "50a0c0ab15ac" + "0" * 52,
}
CAPABILITIES = {
    "qwen3.6:35b-mlx": ["completion", "tools", "thinking"],
    "gemma4:26b-mlx": ["completion", "tools", "vision"],
    "no-tools:latest": ["completion", "thinking"],
    "gemma4:e4b": ["completion", "vision"],
    "x/z-image-turbo:latest": ["image"],
    "x/flux2-klein:latest": ["image"],
}
IMAGE_MODELS = {"x/z-image-turbo:latest", "x/flux2-klein:latest"}


def default_chat(body: dict[str, Any]) -> dict[str, Any]:
    """A well-behaved model: valid JSON, correct tool call, sees red, deterministic."""
    last = body["messages"][-1]
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    if "format" in body:
        message["content"] = ProbeClaim(
            claim="Accuracy improves to 84.6%", evidence_quote="from 82.1% to 84.6%", kind="result"
        ).model_dump_json()
    elif "tools" in body:
        message["tool_calls"] = [
            {"function": {"name": "use_paper_figure", "arguments": {"figure_id": "fig3"}}}
        ]
    elif last.get("images"):
        message["content"] = "Red."
    else:
        message["content"] = f"OK seed={body['options'].get('seed')}"
    return {
        "model": body["model"],
        "message": message,
        "done": True,
        "prompt_eval_count": 20,
        "eval_count": 50,
        "load_duration": 2_000_000_000,
        "prompt_eval_duration": 100_000_000,
        "eval_duration": 1_000_000_000,
        "total_duration": 3_200_000_000,
    }


def default_image(body: dict[str, Any]) -> dict[str, Any]:
    """Deterministic image: colour derived from the seed."""
    seed = body.get("options", {}).get("seed", 0)
    png = solid_png(((seed * 40) % 256, 100, 100), size=8)
    return {
        "model": body["model"],
        "image": base64.b64encode(png).decode(),
        "done": True,
        "total_duration": 5_000_000_000,
    }


class FakeOllama:
    """In-memory stand-in for an Ollama server, served through ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.installed = dict(INSTALLED)
        self.loaded: set[str] = set()
        self.sticky: set[str] = set()  # models that refuse to unload
        self.unload_delay_polls = 0  # /api/ps calls a model stays listed after keep_alive=0
        self._pending_unload: dict[str, int] = {}
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.chat_handler: Callable[[dict[str, Any]], dict[str, Any]] = default_chat
        self.image_handler: Callable[[dict[str, Any]], dict[str, Any]] = default_image
        self.version = "0.24.0"

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.requests.append((path, body))
        model = body.get("model")
        if model is not None and model not in self.installed:
            return httpx.Response(404, json={"error": f"model '{model}' not found"})
        if path == "/api/chat":
            self.loaded.add(model)
            return httpx.Response(200, json=self.chat_handler(body))
        if path == "/api/generate":
            if body.get("keep_alive") == 0 and "prompt" not in body:
                if model in self.loaded and model not in self.sticky:
                    if self.unload_delay_polls:
                        self._pending_unload[model] = self.unload_delay_polls
                    else:
                        self.loaded.discard(model)
                return httpx.Response(200, json={"model": model, "done": True})
            self.loaded.add(model)
            return httpx.Response(200, json=self.image_handler(body))
        if path == "/api/tags":
            models = [{"name": n, "digest": d} for n, d in self.installed.items()]
            return httpx.Response(200, json={"models": models})
        if path == "/api/ps":
            for name, left in list(self._pending_unload.items()):
                if left <= 0:
                    self.loaded.discard(name)
                    del self._pending_unload[name]
                else:
                    self._pending_unload[name] = left - 1
            return httpx.Response(200, json={"models": [{"name": n} for n in sorted(self.loaded)]})
        if path == "/api/show":
            return httpx.Response(200, json={"capabilities": CAPABILITIES.get(model, [])})
        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        return httpx.Response(404, text="not found")

    def client(self) -> OllamaClient:
        return OllamaClient(transport=httpx.MockTransport(self.handle))

    def paths(self) -> list[str]:
        return [p for p, _ in self.requests]


@pytest.fixture
def fake() -> FakeOllama:
    return FakeOllama()


@pytest.fixture(autouse=True)
def fast_unload_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never really sleep while waiting for (fake) unloads."""
    monkeypatch.setattr("paper2carousel.probe.UNLOAD_WAIT_S", 0.2)
    monkeypatch.setattr("paper2carousel.probe.UNLOAD_POLL_S", 0.0)


# --- M1 helpers: synthetic papers, a fake arXiv, a deck-writing model ---------------------


SECTIONS = [
    ("Introduction", "Transformers are everywhere. We study attention cost."),
    ("Method", "We propose LinAttn with linear complexity in sequence length."),
    ("Results", "LinAttn reaches 84.6% accuracy, up from 82.1%, with 38% less memory."),
]


def make_pdf(path, sections=SECTIONS, toc=True, references=True, title="A Test Paper"):
    """Write a small multi-page PDF with numbered headings and (optionally) an outline."""
    doc = pymupdf.open()
    entries = []
    for i, (heading, body) in enumerate(sections, start=1):
        page = doc.new_page()
        page.insert_text((72, 72), f"{i} {heading}", fontsize=14)
        page.insert_text((72, 100), body, fontsize=10)
        entries.append([1, heading, i])
    if references:
        page = doc.new_page()
        page.insert_text((72, 72), "References", fontsize=14)
        page.insert_text((72, 100), "[1] Someone. A cited paper. 2020.", fontsize=10)
    if toc:
        doc.set_toc(entries)
    doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>A Test
      Paper</title>
    <summary>  We study attention.
      It is linear now. </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
  </entry>
</feed>"""

EMPTY_ATOM = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class FakeArxiv:
    """Serves the Atom API and the PDF for any id; counts downloads."""

    def __init__(self, pdf_bytes: bytes, atom: str = ATOM) -> None:
        self.pdf_bytes = pdf_bytes
        self.atom = atom
        self.pdf_downloads = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/query"):
            return httpx.Response(200, text=self.atom)
        if request.url.path.startswith("/pdf/"):
            self.pdf_downloads += 1
            return httpx.Response(200, content=self.pdf_bytes)
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


@pytest.fixture
def arxiv(tmp_path):
    return FakeArxiv(make_pdf(tmp_path / "src.pdf").read_bytes())


DECK = Deck(
    title="Linear attention, same accuracy",
    slides=[
        DraftSlide(title="The problem", bullets=["Attention cost grows quadratically."]),
        DraftSlide(title="The idea", bullets=["LinAttn is linear in sequence length."]),
        DraftSlide(title="Results", bullets=["84.6% accuracy, up from 82.1%.", "38% less memory."]),
    ],
)


def deck_chat(body: dict[str, Any]) -> dict[str, Any]:
    """A model that answers deck requests with DECK and everything else like default_chat."""
    if "slides" in body.get("format", {}).get("properties", {}):
        return {"model": body["model"], "message": {"content": DECK.model_dump_json()}}
    return default_chat(body)
