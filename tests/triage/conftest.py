"""A fake arXiv with five papers and a decision model that rates every one of them."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from labmate.config import CacheConfig, Config, ModelsConfig, TracingConfig
from tests.conftest import FakeArxiv, agentic_chat, atom_feed, make_pdf, reply

TITLES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]


FEED = atom_feed([(f"2401.0000{i}", t) for i, t in enumerate(TITLES, start=1)])


@pytest.fixture
def arxiv_five(tmp_path):
    return FakeArxiv(make_pdf(tmp_path / "src.pdf").read_bytes(), atom=FEED)


@pytest.fixture
def config(tmp_path):
    """A chat model as the decider (see ``decision_config`` for a decision model)."""
    return Config(
        cache=CacheConfig(enabled=False),
        models=ModelsConfig(decider="gemma4:e4b"),
        tracing=TracingConfig(runs_dir=tmp_path / "runs"),
    )


@pytest.fixture
def decision_config(tmp_path):
    return Config(
        cache=CacheConfig(enabled=False),
        models=ModelsConfig(decider="clef-flash"),
        tracing=TracingConfig(runs_dir=tmp_path / "runs"),
    )


def scorer(scores: dict[str, float]):
    """A decision model that scores the paper whose title is in the state."""

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        title = re.search(r"Paper title: (\w+)", body["state"]).group(1)
        return {"answers": {"relevance": {"type": "score", "score": scores[title]}}}

    return handler


def decider(action: str = "deep", reason: str = "Fits the interests.", scores=None):
    """A decision model: rates the paper whose title is in the prompt (default 5, or by title)."""

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        if "action" not in body.get("format", {}).get("properties", {}):
            return agentic_chat(body)  # the paper chain steps
        prompt = body["messages"][-1]["content"]
        title = re.search(r"Title: (\w+)", prompt).group(1)
        relevance = (scores or {}).get(title, 5)
        return reply(body, json.dumps(
            {"action": action, "relevance": relevance, "reason": reason}))  # fmt: skip

    return handler
