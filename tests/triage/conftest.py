"""A fake arXiv with five papers and a decision model that rates every one of them."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from labmate.config import CacheConfig, Config, TracingConfig
from tests.conftest import FakeArxiv, agentic_chat, make_pdf, reply

TITLES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]


def entry(number: int, title: str) -> str:
    return f"""<entry>
    <id>http://arxiv.org/abs/2401.0000{number}v1</id>
    <title>{title}</title>
    <summary>We study {title.lower()}.</summary>
    <author><name>Ada Lovelace</name></author>
    <published>2024-01-02T00:00:00Z</published>
  </entry>"""


FEED = (
    '<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
    + "".join(entry(i, t) for i, t in enumerate(TITLES, start=1))
    + "</feed>"
)


@pytest.fixture
def arxiv_five(tmp_path):
    return FakeArxiv(make_pdf(tmp_path / "src.pdf").read_bytes(), atom=FEED)


@pytest.fixture
def config(tmp_path):
    return Config(
        cache=CacheConfig(enabled=False),
        tracing=TracingConfig(runs_dir=tmp_path / "runs"),
    )


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
