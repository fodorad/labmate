"""A fake arXiv with two papers and a scripted scout agent."""

from __future__ import annotations

from typing import Any

import pytest

from labmate.config import CacheConfig, Config, TracingConfig
from tests.conftest import FakeArxiv, agentic_chat, make_pdf, reply, tool_call


def entry(arxiv_id: str, title: str) -> str:
    return f"""<entry>
    <id>http://arxiv.org/abs/{arxiv_id}v1</id>
    <title>{title}</title>
    <summary>We study {title.lower()}.</summary>
    <author><name>Ada Lovelace</name></author>
    <published>2024-01-02T00:00:00Z</published>
  </entry>"""


FEED = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  {entry("2401.00001", "Linear Attention Revisited")}
  {entry("2401.00002", "Sparse Attention at Scale")}
</feed>"""


@pytest.fixture
def arxiv_two(tmp_path):
    return FakeArxiv(make_pdf(tmp_path / "src.pdf").read_bytes(), atom=FEED)


@pytest.fixture
def config(tmp_path):
    return Config(
        cache=CacheConfig(path=tmp_path / "cache.sqlite"),
        tracing=TracingConfig(runs_dir=tmp_path / "runs"),
    )


def scripted(*calls: tuple[str, dict[str, Any]]):
    """The agent makes these tool calls in order, then stops; other prompts get paper answers."""

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        if not body.get("tools"):
            return agentic_chat(body)
        done = sum(1 for m in body["messages"] if m["role"] == "tool")
        if done < len(calls):
            name, arguments = calls[done]
            return tool_call(body, name, **arguments)
        return reply(body, "That is all.")

    return handler


def tool_results(fake) -> list[str]:
    """Every tool result the agent saw, from its last request."""
    last = [b for p, b in fake.requests if p == "/api/chat" and b.get("tools")][-1]
    return [m["content"] for m in last["messages"] if m["role"] == "tool"]
