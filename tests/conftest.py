"""Shared fixtures: an in-memory fake of the Ollama HTTP API."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx
import pymupdf
import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from labmate.config import CacheConfig, Config
from labmate.core.chat import chat_model

INSTALLED = ("qwen3.6:35b-mlx", "gemma4:26b-mlx", "embeddinggemma:latest")


class SampleClaim(BaseModel):
    """A small schema for structured-output tests."""

    claim: str
    evidence_quote: str
    kind: str


def default_chat(body: dict[str, Any]) -> dict[str, Any]:
    """A well-behaved model: valid JSON, correct tool call, deterministic."""
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    if "format" in body:
        message["content"] = SampleClaim(
            claim="Accuracy improves to 84.6%", evidence_quote="from 82.1% to 84.6%", kind="result"
        ).model_dump_json()
    elif body.get("tools"):
        message["tool_calls"] = [
            {"function": {"name": "use_paper_figure", "arguments": {"figure_id": "fig3"}}}
        ]
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


EMBED_DIM = 64


def default_embed(body: dict[str, Any]) -> dict[str, Any]:
    """Deterministic bag-of-words embeddings: each word (lower-cased, 4+ letters, crude
    plural stripping) adds to a hashed dimension, so texts sharing words are similar."""
    import hashlib

    vectors = []
    for text in body["input"]:
        vector = [0.0] * EMBED_DIM
        for word in re.findall(r"[a-z]{4,}", text.lower()):
            word = word[:-1] if word.endswith("s") else word
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBED_DIM] += 1.0
        vectors.append(vector)
    return {"model": body["model"], "embeddings": vectors, "prompt_eval_count": 7,
            "total_duration": 1_000_000}  # fmt: skip


def reply(body: dict[str, Any], content: str) -> dict[str, Any]:
    """A chat response of the fake server carrying ``content``."""
    return {"model": body["model"], "message": {"role": "assistant", "content": content}}


def tool_call(body: dict[str, Any], name: str, **arguments: Any) -> dict[str, Any]:
    """A chat response in which the model calls one tool."""
    call = {"function": {"name": name, "arguments": arguments}}
    message = {"role": "assistant", "content": "", "tool_calls": [call]}
    return {"model": body["model"], "message": message}


class FakeOllama:
    """In-memory stand-in for an Ollama server, served through ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.chat_handler: Callable[[dict[str, Any]], dict[str, Any]] = default_chat
        self.embed_handler: Callable[[dict[str, Any]], dict[str, Any]] = default_embed

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.requests.append((path, body))
        model = body.get("model")
        if model is not None and model not in INSTALLED:
            return httpx.Response(404, json={"error": f"model '{model}' not found"})
        if path == "/api/chat":
            return httpx.Response(200, json=self.chat_handler(body))
        if path == "/api/embed":
            return httpx.Response(200, json=self.embed_handler(body))
        return httpx.Response(404, text="not found")

    def transport(self) -> httpx.MockTransport:
        """A transport that reaches this fake from ``ChatOllama``."""
        return httpx.MockTransport(self.handle)

    def paths(self) -> list[str]:
        return [p for p, _ in self.requests]


@pytest.fixture
def fake() -> FakeOllama:
    return FakeOllama()


@pytest.fixture(autouse=True)
def no_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never follow the developer's LABMATE_OLLAMA_HOST."""
    monkeypatch.delenv("LABMATE_OLLAMA_HOST", raising=False)


# --- synthetic papers and a fake arXiv ----------------------------------------------------


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
    path,
    sections=SECTIONS,
    toc=True,
    references=True,
    title="A Test Paper",
    figure=False,
    metadata=None,
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
        entries.append([1, "References", doc.page_count])
    if toc:
        doc.set_toc(entries)
    doc.set_metadata({"title": title, **(metadata or {})})
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

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport())


@pytest.fixture
def arxiv(tmp_path):
    return FakeArxiv(make_pdf(tmp_path / "src.pdf").read_bytes())


# --- a fake model that plays every step of the paper chains ---------------------------------

ALL_IDS = re.compile(r"^(c\d{2}) ", re.MULTILINE)


def _last_user(body: dict[str, Any]) -> str:
    return next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")


def _field(prompt: str, name: str) -> str:
    match = re.search(rf"^{name}: (.*)$", prompt, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _first_sentences(text: str, n: int) -> list[str]:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.split()) >= 4]
    return sentences[:n]


def _flow_words(prompt: str) -> list[str]:
    """Distinct words of the section text a flow prompt shows, to use as grounded labels."""
    text = prompt.split("Section text:", 1)[-1]
    return list(dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z]{6,}", text)))


def agentic_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Answer each pipeline step's schema with plausible, grounded content."""
    props = body.get("format", {}).get("properties", {})
    prompt = _last_user(body)
    conversation = "\n".join(m["content"] for m in body["messages"] if m["role"] == "user")
    if "takeaways" in props:
        ids = re.findall(r"\[(c\d{2})", prompt)
        content = {
            "hook": "Linear attention without the accuracy tax",
            "takeaways": [
                {"text": "A grounded takeaway.", "claim_ids": [ids[i % len(ids)]]} for i in range(3)
            ],
            "question": "Where would linear attention help your models?",
        }
    elif "venue" in props:
        content = {"venue": "", "date": ""}
    elif "expand" in props:
        words = _flow_words(prompt)[:4]
        kinds = ["input", "process", "component", "output"]
        content = {
            "title": "The method end to end",
            "nodes": [{"id": f"n{i}", "label": w, "kind": kinds[i]} for i, w in enumerate(words)],
            "edges": [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(len(words) - 1)],
            "caption": "The method in one flow.",
            "expand": ["n1", "n2"],
        }
    elif "nodes" in props:
        words = _flow_words(prompt)[4:7]
        kinds = ["data", "process", "output"]
        content = {
            "title": "Inside one step",
            "nodes": [{"id": f"d{i}", "label": w, "kind": kinds[i]} for i, w in enumerate(words)],
            "edges": [{"source": f"d{i}", "target": f"d{i + 1}"} for i in range(len(words) - 1)],
            "caption": "What happens inside the step.",
        }
    elif "paper_type" in props:
        content: Any = {"paper_type": "method", "confidence": 0.9, "reason": "new model"}
    elif "claims" in props:
        text = prompt.split("SECTION TEXT:", 1)[1]
        section = _field(prompt, "Section")
        kinds = {"Introduction": ["task", "challenge"], "Method": ["method", "contribution"]}
        wanted = kinds.get(section.split(" (")[0], ["result", "result"])
        content = {
            "claims": [
                {"claim": f"Claim: {q}.", "evidence_quote": q, "kind": kind}
                for q, kind in zip(_first_sentences(text, 2), wanted, strict=False)
            ]
        }
    elif "cards" in props:
        ids = ALL_IDS.findall(conversation)
        purposes = ["task", "challenges", "method", "results"]
        content = {
            "cards": [
                {"title": f"Card about {p}", "purpose": p, "claim_ids": [ids[i % len(ids)]]}
                for i, p in enumerate(purposes)
            ],
        }
    elif "icons" in props:
        sentences = re.findall(r"^\d+\. ", prompt, re.MULTILINE)
        content = {"icons": [["target", "bulb", "cpu", "chart-bar", "rocket"][i % 5]
                             for i in range(len(sentences))]}  # fmt: skip
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
    elif "bullets" in props:
        ids = ALL_IDS.findall(conversation)
        content = {
            "title": "A written card",
            "bullets": [
                {"text": "A grounded bullet on the architecture drawing.", "claim_ids": [cid]}
                for cid in ids
            ],
        }
    else:
        return default_chat(body)
    return {
        "model": body["model"],
        "message": {"role": "assistant", "content": json.dumps(content)},
    }


def chat(fake: FakeOllama, model: str = "qwen3.6:35b-mlx") -> BaseChatModel:
    """A chat model talking to the fake server, with the reply cache off."""
    return chat_model(Config(cache=CacheConfig(enabled=False)), model, transport=fake.transport())
