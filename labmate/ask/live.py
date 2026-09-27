"""What the dashboard shows, computed from the graph's stream and the trace (no UI code).

The dashboard streams the agent with ``stream_mode=["updates", "custom"]`` and
``subgraphs=True``. :class:`LiveView` folds those events into: which nodes have run
(highlighted on the Mermaid overview), a readable event log, the latest retrieved chunks,
a pending clarification, and the answer. :func:`model_stats` summarises the model calls
from the trace.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from labmate.ask.schemas import Answer
from labmate.diagrams import ASK_OVERVIEW

SUBGRAPH_NODES = {
    "research": {"retrieve", "grade", "rewrite"},
    "verify": {"judge", "rewrite"},
}
"""Subgraph nodes that appear on the overview (as ``<subgraph>_<node>``)."""

DONE_STYLE = "fill:#e4f8d6,stroke:#6fa24c"
ACTIVE_STYLE = "fill:#f1e1dc,stroke:#ab4c31,stroke-width:3px"


class ModelStats(BaseModel):
    """Model calls seen in the trace.

    Attributes:
        calls: Chat calls per model.
        embeds: Embedding calls.
        tokens_in: Prompt tokens.
        tokens_out: Generated tokens.
        cached: Calls answered from cassettes.
        last_model: Model of the most recent call.
        seconds: Time spent in model calls.
    """

    calls: dict[str, int] = Field(default_factory=dict)
    embeds: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached: int = 0
    last_model: str = ""
    seconds: float = 0.0


def model_stats(spans: Iterable[dict[str, Any]]) -> ModelStats:
    """Summarise the ``llm.*`` spans of a trace.

    Args:
        spans: Trace spans.

    Returns:
        The summary.
    """
    stats = ModelStats()
    for span in spans:
        name = span.get("name")
        if name not in ("llm.chat", "llm.embed"):
            continue
        model = str(span.get("model", ""))
        stats.last_model = model
        stats.cached += bool(span.get("cached"))
        stats.tokens_in += int(span.get("tokens_in") or 0)
        stats.seconds += float(span.get("latency_ms") or 0) / 1000
        if name == "llm.embed":
            stats.embeds += 1
            continue
        stats.calls[model] = stats.calls.get(model, 0) + 1
        stats.tokens_out += int(span.get("tokens_out") or 0)
    return stats


def node_id(namespace: Sequence[str], node: str) -> str | None:
    """The overview's id for a node that just ran, if it is drawn.

    Args:
        namespace: Stream namespace (empty for the top-level graph, ``("research:<id>",)``
            inside a subgraph).
        node: Node name.

    Returns:
        The Mermaid node id, or ``None`` for nodes the overview leaves out.
    """
    if not namespace:
        return node
    parent = namespace[-1].split(":", 1)[0]
    return f"{parent}_{node}" if node in SUBGRAPH_NODES.get(parent, set()) else None


def describe(event: dict[str, Any]) -> str:
    """One log line for a custom event.

    Args:
        event: A custom stream event.

    Returns:
        Readable text.
    """
    kind = event.get("event")
    if kind == "understood":
        options = f"; ambiguous: {', '.join(event['options'])}" if event.get("options") else ""
        return f"understood as {event['intent']}: {event['standalone']}{options}"
    if kind == "planned":
        return "planned " + " | ".join(event["queries"])
    if kind == "retrieved":
        n, tiers = len(event["hits"]), event["tiers"]
        return f"retrieved {n} chunks (tiers {tiers}) for: {event['query']}"
    if kind == "graded":
        verdict = "enough" if event["sufficient"] else f"missing: {event.get('missing') or '?'}"
        return f"graded {len(event['relevant'])} relevant, {verdict}"
    if kind == "rewritten":
        return f"rewrote the query (tiers {event['tiers']}): {event['query']}"
    if kind == "conflicts":
        return f"conflicts: {len(event['items'])}"
    if kind == "drafted":
        return f"drafted {len(event['sentences'])} sentences"
    if kind == "verified":
        return f"verified: kept {event['kept']}, dropped {event['dropped']}"
    return str(event)


class LiveView:
    """State of one question as it streams.

    Attributes:
        done: Overview node ids that have run, in order.
        log: Readable events.
        hits: Chunks of the latest retrieval.
        interrupt: Pending clarification (``{"question", "options"}``), if any.
        answer: The final answer, once there is one.
    """

    def __init__(self) -> None:
        self.done: list[str] = []
        self.log: list[str] = []
        self.hits: list[dict[str, Any]] = []
        self.interrupt: dict[str, Any] | None = None
        self.answer: Answer | None = None

    def feed(self, item: tuple[Sequence[str], str, Any]) -> None:
        """Take one stream item ``(namespace, mode, chunk)``.

        Args:
            item: From ``graph.stream(..., stream_mode=["updates", "custom"],
                subgraphs=True)``.
        """
        namespace, mode, chunk = item
        if mode == "custom":
            self.log.append(describe(chunk))
            if chunk.get("event") == "retrieved":
                self.hits = list(chunk["hits"])
            return
        for node, update in chunk.items():
            if node == "__interrupt__":
                self.interrupt = dict(update[0].value)
                self.log.append("waiting for a choice: " + " | ".join(self.interrupt["options"]))
                continue
            nid = node_id(namespace, node)
            if nid is not None:
                self.done.append(nid)
            if isinstance(update, dict) and isinstance(update.get("answer"), Answer):
                self.answer = update["answer"]

    def mermaid(self, base: str = ASK_OVERVIEW) -> str:
        """The overview with run nodes in green and the latest one outlined.

        Args:
            base: Mermaid flowchart whose node ids match :func:`node_id`.

        Returns:
            Mermaid source.
        """
        lines = [
            base.rstrip(),
            f"    classDef done {DONE_STYLE}",
            f"    classDef active {ACTIVE_STYLE}",
        ]
        seen = list(dict.fromkeys(self.done))
        if seen[:-1]:
            lines.append(f"    class {','.join(seen[:-1])} done")
        if seen:
            lines.append(f"    class {seen[-1]} active")
        return "\n".join(lines) + "\n"
