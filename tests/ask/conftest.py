"""Fixtures for ask: a small library (dissertation, paper, outside paper) and a fake model
that plays every role of the agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from labmate.ask.session import open_ask
from labmate.config import Config
from tests.conftest import agentic_chat

DISSERTATION = [
    (1, "1 Introduction", 1,
     "Eye blink detection supports driver drowsiness monitoring and deepfake detection. "
     "Frame-wise eye state classifiers struggle with short blinks and noisy video. This "
     "dissertation studies multimodal transformers for affective computing tasks such as "
     "blink detection and emotion recognition."),
    (1, "2 Method", 2,
     "This chapter introduces the proposed linear multimodal transformer and its training."),
    (2, "2.1 Architecture", 2,
     "BlinkLinMulT fuses RGB texture and iris landmarks with linear attention. Cross-modal "
     "transformers translate landmark features into the texture embedding space. The model "
     "reaches 0.912 F1 on RT-BENE when trained on the union of datasets.\n"
     "Figure 1: The BlinkLinMulT architecture with its cross-modal branches."),
    (2, "2.2 Training", 3,
     "Training uses the union of CEW, ZJU and MRL Eye datasets. Heavy augmentation with "
     "lighting changes improves robustness to head pose variation. Each model trains for "
     "forty epochs on a single graphics card."),
    (1, "References", 5, "[1] Someone. A cited paper. 2020."),
]  # fmt: skip

PAPER = [
    (1, "1 Results", 1,
     "BlinkLinMulT reaches 0.905 F1 on RT-BENE in the journal version of the experiments. "
     "The ablation shows that iris landmarks contribute most of the gain over texture alone. "
     "Results are averaged over five runs with different random seeds."),
    (1, "2 Datasets", 2,
     "The evaluation uses the CEW, ZJU, MRL Eye, RT-BENE, EyeBlink8 and TalkingFace datasets. "
     "Each dataset is split into training and test partitions following earlier work, and "
     "the splits are published with the code."),
]  # fmt: skip

OUTSIDE = [
    (1, "1 Background", 1,
     "Transformers use attention over tokens to model long sequences in language tasks. "
     "Attention cost grows quadratically with sequence length, which limits long inputs, "
     "and many efficient variants approximate it."),
]  # fmt: skip


def make_doc(path: Path, sections: list[tuple[int, str, int, str]]) -> Path:
    """A PDF with a multi-level outline; ``sections`` are (level, heading, page, body)."""
    doc = pymupdf.open()
    toc, y = [], {}
    for level, heading, page, body in sections:
        while doc.page_count < page:
            doc.new_page()
            y[doc.page_count] = 72
        pg = doc[page - 1]
        top = y[page]
        pg.insert_text((72, top), heading, fontsize=13 if level == 1 else 11)
        rect = pymupdf.Rect(72, top + 12, 520, top + 250)
        pg.insert_textbox(rect, body, fontsize=9)
        y[page] = top + 260
        toc.append([level, heading.split(" ", 1)[1] if heading[0].isdigit() else heading, page])
    doc.set_toc(toc)
    doc.save(path)
    doc.close()
    return path


LIBRARY_TOML = """
[[source]]
id = "dissertation"
file = "dissertation.pdf"
title = "Multimodal transformers for affective computing"
label = "Dissertation"
year = 2025

[[source]]
id = "blinklinmult"
file = "papers/blinklinmult.pdf"
title = "BlinkLinMulT: Transformer-based Eye Blink Detection"
label = "BlinkLinMulT"
year = 2023

[[source]]
id = "outside"
file = "outside.pdf"
title = "Attention over tokens"
label = "Outside"
"""


@pytest.fixture
def library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    (root / "papers").mkdir(parents=True)
    make_doc(root / "dissertation.pdf", DISSERTATION)
    make_doc(root / "papers" / "blinklinmult.pdf", PAPER)
    make_doc(root / "outside.pdf", OUTSIDE)
    (root / "library.toml").write_text(LIBRARY_TOML)
    return root


@pytest.fixture
def config(tmp_path: Path, library: Path) -> Config:
    cfg = Config()
    cfg.cache.path = tmp_path / "cache" / "replies.sqlite"
    cfg.ask.library = library
    cfg.ask.index = library / "index.sqlite"
    cfg.ask.chunk_words = 30
    cfg.ask.top_k = 4
    return cfg


# --- the fake model -----------------------------------------------------------------------

_IDS = re.compile(r"^\[([a-z0-9-]+:\d{4})\]", re.MULTILINE)


def _field(prompt: str, name: str) -> str:
    match = re.search(rf"^{name}: (.*)$", prompt, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _reply(body: dict[str, Any], content: Any) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content)
    return {"model": body["model"], "message": {"role": "assistant", "content": text},
            "prompt_eval_count": 30, "eval_count": 10}  # fmt: skip


def _evidence(prompt: str) -> dict[str, str]:
    """Chunk id -> text of an evidence block in a prompt."""
    blocks = re.split(
        r"\n\n(?=\[)", prompt.split("Evidence:", 1)[-1].split("Search results:", 1)[-1]
    )
    out = {}
    for block in blocks:
        m = re.match(r"\s*\[([a-z0-9-]+:\d{4})\][^\n]*\n(.*)", block, re.DOTALL)
        if m:
            out[m.group(1)] = " ".join(m.group(2).split())
    return out


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{5,}", text.lower())}


def ask_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Answers every ask prompt deterministically; delegates the rest to agentic_chat."""
    props = (
        body.get("format", {}).get("properties", {}) if isinstance(body.get("format"), dict) else {}
    )
    user = [m for m in body["messages"] if m["role"] == "user"]
    prompt = user[0]["content"] if user else ""  # the task (later user turns are feedback)
    if body.get("tools"):
        return _agent_turn(body)
    if "on_topic" in props:
        question = _field(prompt, "Question")
        if "weather" in question.lower():
            return _reply(body, {"on_topic": False, "options": [], "search": question})
        if "the transformer" in question.lower():
            options = ["BlinkLinMulT", "the outside transformer"]
            return _reply(body, {"on_topic": True, "options": options, "search": question})
        return _reply(body, {"on_topic": True, "options": [], "search": question})
    if "sufficient" in props:
        query = _field(prompt, "Search query")
        shown = _evidence(prompt)
        wanted = _words(query) - {"about", "which", "there"}
        relevant = [cid for cid, text in shown.items() if wanted & _words(text)]
        sufficient = bool(relevant) and "rarely" not in query
        return _reply(body, {"relevant": relevant, "sufficient": sufficient,
                             "missing": "" if sufficient else "more detail"})  # fmt: skip
    if set(props) == {"query"}:
        last = _field(prompt, "Last query")
        return _reply(body, {"query": last.replace("rarely", "").strip() + " details"})
    if "verdicts" in props and "Sentences of the answer" in prompt:
        numbered = re.findall(r"^Sentence (\d+): (.*)$", prompt, re.MULTILINE)
        return _reply(body, {"verdicts": [{"sentence": int(n), "supported": "BOGUS" not in text}
                                          for n, text in numbered]})  # fmt: skip
    if "sentences" in props:
        shown = _evidence(prompt)
        sentences = []
        for cid, text in list(shown.items())[:2]:
            first = re.split(r"(?<=\.)\s", text)[0]
            words = first.split()[:20]
            sentences.append({"text": " ".join(words).rstrip(".") + ".", "chunk_ids": [cid]})
        if "BOGUS" in prompt:  # right words, wrong attribution: only the judge can tell
            sentences.append({"text": "BOGUS: the model trains for forty epochs on CEW.",
                              "chunk_ids": [next(iter(shown))]})  # fmt: skip
        if "WRONG" in prompt:
            sentences.append({"text": "It reaches 99.9 F1 on every benchmark.",
                              "chunk_ids": [next(iter(shown))]})  # fmt: skip
        return _reply(body, {"sentences": sentences})
    return agentic_chat(body)


def _agent_turn(body: dict[str, Any]) -> dict[str, Any]:
    """The agent: search once, then answer citing the first hit."""
    tool_results = [m for m in body["messages"] if m["role"] == "tool"]
    question = next(m["content"] for m in body["messages"] if m["role"] == "user")
    if not tool_results:
        call = {"function": {"name": "search_library", "arguments": {"query": question}}}
        return {"model": body["model"],
                "message": {"role": "assistant", "content": "", "tool_calls": [call]}}  # fmt: skip
    ids = _IDS.findall(tool_results[-1]["content"])
    if not ids:
        return _reply(body, "The library does not answer that.")
    text = tool_results[-1]["content"].split(": ", 1)[1].split("\n")[0]
    first = " ".join(text.split()[:12]).rstrip(".")
    return _reply(body, f"{first} [{ids[0]}]. Unsupported claim without a citation.")


@pytest.fixture
def model(fake):
    fake.chat_handler = ask_chat
    return fake


@pytest.fixture
def session(config, model):
    s = open_ask(config, ollama=model.transport())
    yield s
    s.close()


@pytest.fixture
def indexed(session, library):
    from labmate.ask.build import build_index
    from labmate.ask.library import load_library

    build_index(session, load_library(library))
    return session
