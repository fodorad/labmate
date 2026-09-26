"""Shared fixtures: an in-memory fake of the Ollama HTTP API."""

from __future__ import annotations

import base64
import json
import re
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
    (
        "Introduction",
        "Transformers are everywhere in modern machine learning systems. Their attention "
        "cost grows quadratically with the sequence length, which limits long inputs. "
        "We study how to remove this bottleneck without losing accuracy on benchmarks.",
    ),
    (
        "Method",
        "We propose LinAttn with linear complexity in sequence length. It replaces the "
        "softmax kernel with a feature map that can be computed incrementally. The model "
        "keeps the same number of layers and parameters as the baseline transformer.",
    ),
    (
        "Results",
        "LinAttn reaches 84.6% accuracy, up from 82.1%, with 38% less memory. Training is "
        "twice as fast on sequences of length 4096. The gains grow with sequence length "
        "while short sequences show no measurable difference.",
    ),
]


def make_pdf(
    path, sections=SECTIONS, toc=True, references=True, title="A Test Paper", figure=False
):
    """Write a small multi-page PDF with numbered headings and (optionally) an outline.

    With ``figure=True`` the first page also gets a raster image with a "Figure 1:" caption
    and a vector drawing with a "Figure 2:" caption.
    """
    doc = pymupdf.open()
    entries = []
    for i, (heading, body) in enumerate(sections, start=1):
        page = doc.new_page()
        page.insert_text((72, 72), f"{i} {heading}", fontsize=14)
        page.insert_textbox(pymupdf.Rect(72, 90, 520, 300), body, fontsize=10)
        if figure and i == 1:
            png = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 80), False)
            png.set_rect(png.irect, (40, 90, 200))
            page.insert_text((150, 318), "Encoder", fontsize=9)
            page.insert_image(pymupdf.Rect(150, 320, 390, 480), pixmap=png)
            page.insert_text((72, 500), "Figure 1: A synthetic architecture diagram.", fontsize=9)
            shape = page.new_shape()
            shape.draw_rect(pymupdf.Rect(150, 530, 390, 680))
            shape.draw_line((150, 530), (390, 680))
            shape.finish(color=(0, 0, 0), width=1)
            shape.commit()
            page.insert_text((72, 700), "Figure 2: Vector-only drawing.", fontsize=9)
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


# --- M2: a fake model that plays router, extractor, planner and writer ---------------------

ALL_IDS = re.compile(r"^(c\d{2}) ", re.MULTILINE)


def _last_user(body: dict[str, Any]) -> str:
    return next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")


def _first_sentences(text: str, n: int) -> list[str]:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.split()) >= 4]
    return sentences[:n]


def agentic_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Answer each pipeline step's schema with plausible, grounded content."""
    props = body.get("format", {}).get("properties", {})
    prompt = _last_user(body)
    conversation = "\n".join(m["content"] for m in body["messages"] if m["role"] == "user")
    if "takeaways" in props:
        ids = re.findall(r"\[(c\d{2})", prompt)
        content = {
            "hook": "Linear attention without the accuracy tax",
            "takeaways": [{"text": "A grounded takeaway.", "claim_ids": [cid]} for cid in ids[:3]],
            "question": "Where would linear attention help your models?",
        }
    elif "legible" in props:
        image_seen = bool(body["messages"][-1].get("images"))
        content = {
            "legible": image_seen,
            "overflow": False,
            "visual_relevant": "UNRELATED" not in prompt,
            "alt_text": "A slide with a title and bullet points." if image_seen else "(no image)",
        }
    elif "paper_type" in props:
        content: Any = {"paper_type": "method", "confidence": 0.9, "reason": "new model"}
    elif "claims" in props:
        text = prompt.split("SECTION TEXT:", 1)[1]
        content = {
            "claims": [
                {"claim": f"Claim: {q}.", "evidence_quote": q, "kind": "result"}
                for q in _first_sentences(text, 2)
            ]
        }
    elif "hook" in props:
        ids = ALL_IDS.findall(conversation)
        purposes = ["problem", "idea", "result", "takeaway"]
        content = {
            "hook": "Linear attention, same accuracy",
            "slides": [
                {"title": f"Slide about {p}", "purpose": p, "claim_ids": [ids[i % len(ids)]]}
                for i, p in enumerate(purposes)
            ],
        }
    elif "verdicts" in props:
        bullets = re.findall(r"^Bullet (\d+): (.*)$", prompt, re.MULTILINE)
        content = {
            "verdicts": [
                {
                    "bullet": int(n),
                    "verdict": "unsupported" if "WRONG" in text else "supported",
                    "reason": "made up" if "WRONG" in text else "stated in the evidence",
                }
                for n, text in bullets
            ]
        }
    elif any(t["function"]["name"] == "no_visual" for t in body.get("tools", [])):
        figure = re.search(r"^(fig\d+) \(page", prompt, re.MULTILINE)
        call = (
            {"name": "use_paper_figure", "arguments": {"figure_id": figure.group(1)}}
            if figure
            else {"name": "no_visual", "arguments": {"reason": "text is clearer"}}
        )
        return {
            "model": body["model"],
            "message": {"content": "", "tool_calls": [{"function": call}]},
        }
    elif "bullets" in props:
        ids = ALL_IDS.findall(conversation)
        content = {
            "title": "A written slide",
            "bullets": [{"text": "A grounded bullet.", "claim_ids": [cid]} for cid in ids],
        }
    else:
        return default_chat(body)
    return {"model": body["model"], "message": {"content": json.dumps(content)}}
