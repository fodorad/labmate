"""Flow diagrams (ORCHESTRATOR–WORKERS): the end-to-end data flow, then its details.

The planner draws the overview, from the raw data to the target output, and picks the
steps that deserve a closer look; one worker call per picked step draws its detail
diagram. The model only proposes graphs as data (typed boxes and arrows). Code checks
them — the flow must start at inputs and end at outputs, every label must name something
the paper says, no number may be invented — and feeds problems back. Code then writes
each graph as Mermaid in a fixed house style and renders it locally with mermaid-cli.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from labmate.core.factcheck import numbers_in
from labmate.core.lc import batch_map, prompt, structured
from labmate.core.structured import StructuredOutputError
from labmate.core.words import content_words
from labmate.paper2flow.prompts import load_prompt
from labmate.paper2flow.schemas import (
    ClaimCard,
    FlowDetail,
    FlowGraph,
    FlowOverview,
    Flows,
    Outline,
    Paper,
)

log = logging.getLogger(__name__)

MERMAID_CLI = ("npx", "--yes", "@mermaid-js/mermaid-cli@12.0.0")
"""The pinned mermaid-cli (``make install`` downloads it once; it runs a headless Chromium)."""

MERMAID_TIMEOUT_S = 120
"""Render timeout for one mermaid-cli call (all diagrams of a paper at once)."""

SCALE = 2
"""Pixel density of the diagram PNGs (2 = sharp on high-DPI screens and in print)."""

MAX_LABEL_WORDS = 5
"""Longest node label."""

MAX_EDGE_WORDS = 4
"""Longest edge label."""

MAX_DETAIL_NODES = 8
"""A detail diagram zooms into one step; more boxes than this is a second overview."""

MAX_EVIDENCE_WORDS = 3000
"""Section text shown to the planner (the context window also holds the claims)."""

LETTERS = "ABC"
"""Names of the detail diagrams, in the order their steps appear in the overview."""

KIND_STYLE = {
    "input": ("#fff8d5", "#c9ad3f"),
    "data": ("#f7f6f4", "#9aa5b1"),
    "component": ("#e3f2f8", "#5f95b0"),
    "process": ("#f1e1dc", "#ab4c31"),
    "output": ("#e4f8d6", "#6fa24c"),
}
"""Fill and border per node kind (the project-page card colours)."""

KIND_NAMES = {
    "input": "Input",
    "data": "Data",
    "component": "Model component",
    "process": "Process",
    "output": "Output",
}
"""Legend entries."""

MERMAID_THEME = {
    "theme": "base",
    "themeVariables": {
        "fontFamily": "Inter, Helvetica, Arial, sans-serif",
        "fontSize": "16px",
        "lineColor": "#4c6176",
        "primaryTextColor": "#222b35",
        "edgeLabelBackground": "#fffefd",
    },
    "flowchart": {"curve": "basis", "nodeSpacing": 40, "rankSpacing": 45, "padding": 14},
}
"""mermaid-cli configuration: the house style shared by every diagram."""

GOAL = (
    "how data flows through the paper's pipeline: from the raw input data, through "
    "preprocessing and the model's components, to the target output"
)
"""What the overview diagram shows."""

# --- evidence ------------------------------------------------------------------------------


def flow_cards(outline: Outline, cards: list[ClaimCard]) -> list[ClaimCard]:
    """Claim cards the diagrams may draw on: the task and method cards, then all method claims.

    Args:
        outline: The outline.
        cards: All claim cards.

    Returns:
        Cards in that order, without duplicates.
    """
    by_id = {c.id: c for c in cards}
    ids = [cid for c in outline.cards if c.purpose in ("task", "method") for cid in c.claim_ids]
    ids += [c.id for c in cards if c.kind in ("method", "contribution")]
    return [by_id[i] for i in dict.fromkeys(ids) if i in by_id]


def flow_sections(paper: Paper, cards: list[ClaimCard], max_words: int = MAX_EVIDENCE_WORDS) -> str:
    """Text of the sections the cards come from, most-cited first, within a word budget.

    Claim quotes are short; the section text is where the pipeline's steps, tensors and
    datasets are actually described.

    Args:
        paper: The paper.
        cards: Cards from :func:`flow_cards`.
        max_words: Budget; a section that doesn't fit is cut at the budget.

    Returns:
        ``## <section>`` blocks of text.
    """
    counts = Counter(c.section for c in cards)
    by_title = {s.title: s for s in paper.sections}
    parts: list[str] = []
    left = max_words
    for title, _ in counts.most_common():
        section = by_title.get(title)
        if section is None or left <= 0:
            continue
        words = section.text.split()[:left]
        left -= len(words)
        parts.append(f"## {title}\n{' '.join(words)}")
    return "\n\n".join(parts)


# --- checks --------------------------------------------------------------------------------


def _norm(label: str) -> str:
    return " ".join(label.lower().split())


def check_graph(
    graph: FlowGraph,
    evidence: str,
    exempt: frozenset[str],
    sink_kinds: tuple[str, ...] = ("output",),
    max_nodes: int = 10,
) -> list[str]:
    """Structure and grounding rules shared by overview and detail diagrams.

    Args:
        graph: Proposed diagram.
        evidence: Text the labels may draw on.
        exempt: Words that need no evidence (the paper title's).
        sink_kinds: Kinds allowed where the flow ends.
        max_nodes: Most boxes allowed.

    Returns:
        Problems; empty if the diagram is valid.
    """
    problems: list[str] = []
    ids = [n.id for n in graph.nodes]
    if len(set(ids)) != len(ids):
        problems.append("node ids must be unique")
    if len(ids) > max_nodes:
        problems.append(f"use at most {max_nodes} nodes, not {len(ids)}")
    known = set(ids)
    for e in graph.edges:
        if e.source not in known or e.target not in known:
            problems.append(f"edge {e.source} -> {e.target} uses an unknown node id")
        elif e.source == e.target:
            problems.append(f"edge {e.source} -> {e.target} points at itself")
        if len(e.label.split()) > MAX_EDGE_WORDS:
            problems.append(f"edge label {e.label!r} has more than {MAX_EDGE_WORDS} words")
    linked = {e.source for e in graph.edges} | {e.target for e in graph.edges}
    problems += [f"node {n} has no edge" for n in ids if n not in linked]
    has_in = {e.target for e in graph.edges}
    has_out = {e.source for e in graph.edges}
    for node in graph.nodes:
        if node.id not in linked:
            continue
        if node.id not in has_in and node.kind not in ("input", "data"):
            problems.append(
                f"the flow starts at {node.label!r}, a {node.kind}; a flow starts at the "
                "input data (kind input or data), so add what goes into it"
            )
        if node.id not in has_out and node.kind not in sink_kinds:
            problems.append(
                f"the flow ends at {node.label!r}, a {node.kind}; it must end at "
                f"{' or '.join(sink_kinds)}, so add what it produces"
            )
    words = content_words(evidence, split_hyphens=True) | exempt
    numbers = numbers_in(evidence)
    problems += [
        f"label {n.label!r} has more than {MAX_LABEL_WORDS} words"
        for n in graph.nodes
        if len(n.label.split()) > MAX_LABEL_WORDS
    ]
    for label in [n.label for n in graph.nodes] + [e.label for e in graph.edges if e.label]:
        if content_words(label, split_hyphens=True) and not (
            content_words(label, split_hyphens=True) & words
        ):
            problems.append(f"label {label!r} names nothing in the evidence")
        if numbers_in(label) - numbers:
            problems.append(f"label {label!r} has a number that is not in the evidence")
    return problems


def check_overview(overview: FlowOverview, evidence: str, exempt: frozenset[str]) -> list[str]:
    """Rules for the overview: :func:`check_graph` plus valid steps to expand.

    Args:
        overview: Proposed overview.
        evidence: Text the labels may draw on.
        exempt: Words that need no evidence.

    Returns:
        Problems; empty if valid.
    """
    problems = check_graph(overview, evidence, exempt)
    kinds = {n.id: n.kind for n in overview.nodes}
    for nid in overview.expand:
        if nid not in kinds:
            problems.append(f"expand names unknown node id {nid!r}")
        elif kinds[nid] not in ("component", "process"):
            problems.append(f"expand {nid!r}: only a component or process step has insides")
    if len(set(overview.expand)) != len(overview.expand):
        problems.append("expand lists a node twice")
    if not overview.expand and len(overview.nodes) >= 4:
        problems.append("expand 1 to 3 steps whose insides the paper describes")
    return problems


def check_detail(
    detail: FlowGraph, overview: FlowOverview, evidence: str, exempt: frozenset[str]
) -> list[str]:
    """Rules for a detail diagram: :func:`check_graph` plus it must add new boxes.

    Args:
        detail: Proposed detail diagram.
        overview: The overview it zooms into.
        evidence: Text the labels may draw on.
        exempt: Words that need no evidence.

    Returns:
        Problems; empty if valid.
    """
    problems = check_graph(
        detail, evidence, exempt, sink_kinds=("output", "data"), max_nodes=MAX_DETAIL_NODES
    )
    known = {_norm(n.label) for n in overview.nodes}
    repeated = [n.label for n in detail.nodes if _norm(n.label) in known]
    if len(repeated) * 2 > len(detail.nodes):
        problems.append(
            f"{len(repeated)} of {len(detail.nodes)} boxes repeat the overview "
            f"({', '.join(repeated)}); show the steps inside this one"
        )
    return problems


# --- planning ------------------------------------------------------------------------------


def _describe(graph: FlowGraph) -> str:
    labels = {n.id: n.label for n in graph.nodes}
    nodes = "\n".join(f"- {n.id}: {n.label} ({n.kind})" for n in graph.nodes)
    edges = "\n".join(
        f"- {labels.get(e.source, e.source)} -> {labels.get(e.target, e.target)}"
        + (f" ({e.label})" if e.label else "")
        for e in graph.edges
    )
    return f"Boxes:\n{nodes}\nArrows:\n{edges}"


def _evidence(cards: list[ClaimCard], sections: str) -> str:
    quotes = "\n".join(f'- "{c.evidence_quote}"' for c in cards) or "(none)"
    return f"Quotes:\n{quotes}\n\nSection text:\n{sections or '(none)'}"


def plan_overview(
    paper: Paper,
    bullets: list[str],
    cards: list[ClaimCard],
    sections: str,
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> FlowOverview:
    """Ask the planner for the end-to-end flow and the steps to expand.

    Args:
        paper: The paper.
        bullets: The fact-checked method card's bullets.
        cards: Evidence cards (:func:`flow_cards`).
        sections: Evidence text (:func:`flow_sections`).
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        An overview that passes :func:`check_overview`.
    """
    evidence = " ".join([*(c.evidence_quote for c in cards), *bullets, sections])
    exempt = frozenset(content_words(paper.title, split_hyphens=True))
    planner = structured(model, FlowOverview, check=lambda g: check_overview(g, evidence, exempt))
    return (prompt(load_prompt("flow_overview")) | planner).invoke(
        {
            "title": paper.title,
            "goal": GOAL,
            "bullets": "\n".join(f"- {b}" for b in bullets) or "(none)",
            "evidence": _evidence(cards, sections),
        },
        config,
    )


def plan_detail(
    paper: Paper,
    overview: FlowOverview,
    node_id: str,
    cards: list[ClaimCard],
    sections: str,
    model: BaseChatModel,
    config: RunnableConfig | None = None,
) -> FlowDetail:
    """Ask a worker for the detail diagram of one overview step.

    Args:
        paper: The paper.
        overview: The overview.
        node_id: The step to break down.
        cards: Evidence cards.
        sections: Evidence text.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        The detail diagram, passing :func:`check_detail`.
    """
    evidence = " ".join([*(c.evidence_quote for c in cards), sections])
    exempt = frozenset(content_words(paper.title, split_hyphens=True))
    labels = {n.id: n.label for n in overview.nodes}
    incoming = [labels[e.source] for e in overview.edges if e.target == node_id]
    outgoing = [labels[e.target] for e in overview.edges if e.source == node_id]
    worker = structured(
        model, FlowGraph, check=lambda g: check_detail(g, overview, evidence, exempt)
    )
    graph = (prompt(load_prompt("flow_detail")) | worker).invoke(
        {
            "title": paper.title,
            "overview": _describe(overview),
            "step": labels[node_id],
            "incoming": ", ".join(incoming) or "(nothing)",
            "outgoing": ", ".join(outgoing) or "(nothing)",
            "evidence": _evidence(cards, sections),
        },
        config,
    )
    return FlowDetail(node_id=node_id, graph=graph)


def plan_flows(
    paper: Paper,
    bullets: list[str],
    cards: list[ClaimCard],
    model: BaseChatModel,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> Flows:
    """The overview (orchestrator), then one detail diagram per picked step (workers).

    Args:
        paper: The paper.
        bullets: The fact-checked method card's bullets.
        cards: Evidence cards (:func:`flow_cards`).
        model: The writer model.
        workers: Concurrent worker calls.
        config: The calling step's config (callbacks).

    Returns:
        All diagrams, details in overview order. A step whose detail diagram stays invalid
        after the retries (the paper hardly describes it) is dropped from the overview's
        ``expand`` list, so the other diagrams still ship.
    """
    sections = flow_sections(paper, cards)
    overview = plan_overview(paper, bullets, cards, sections, model, config)

    def worker(node_id: str, cfg: RunnableConfig) -> FlowDetail | None:
        try:
            return plan_detail(paper, overview, node_id, cards, sections, model, cfg)
        except StructuredOutputError as e:
            log.warning("no detail diagram for %r: %s", node_id, e)
            return None

    results = batch_map(worker, detail_order(overview), config, workers)
    details = [d for d in results if d is not None]
    kept = [d.node_id for d in details]
    overview = overview.model_copy(update={"expand": [n for n in overview.expand if n in kept]})
    return assemble_flows(overview, details)


def detail_order(overview: FlowOverview) -> list[str]:
    """The steps to expand, in the order they appear in the overview (A, B, C).

    Args:
        overview: The overview.

    Returns:
        Node ids.
    """
    order = [n.id for n in overview.nodes]
    return sorted(set(overview.expand), key=order.index)[: len(LETTERS)]


def assemble_flows(overview: FlowOverview, details: list[FlowDetail]) -> Flows:
    """Put the overview and its details together, details in overview order.

    Args:
        overview: The overview.
        details: Detail diagrams (any order).

    Returns:
        The flows.
    """
    by_node = {d.node_id: d for d in details}
    return Flows(overview=overview, details=[by_node[n] for n in detail_order(overview)])


# --- rendering -----------------------------------------------------------------------------


def _text(label: str) -> str:
    """A label as Mermaid text: quotes and angle brackets as entity codes, never markup."""
    return (
        label.replace("#", "#35;").replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")
    )


def flow_mermaid(graph: FlowGraph, markers: dict[str, str] | None = None) -> str:
    """Mermaid source in the house style, top to bottom (built by code, never by the model).

    Node ids are renumbered (``n0``, ``n1``, ...), so a model's id can't clash with a
    Mermaid keyword, and labels are escaped, so paper text can't inject Mermaid syntax.

    Args:
        graph: Flow diagram.
        markers: Node id to detail letter; those boxes get a "detail A" note and a thick
            border, pointing to their detail diagram.

    Returns:
        Mermaid source.
    """
    markers = markers or {}
    ids = {n.id: f"n{i}" for i, n in enumerate(graph.nodes)}
    lines = ["flowchart TB"]
    for kind, (fill, border) in KIND_STYLE.items():
        lines.append(f"  classDef {kind} fill:{fill},stroke:{border},stroke-width:1.6px")
    lines.append("  classDef expanded stroke-width:4px")
    for node in graph.nodes:
        text = _text(node.label)
        if node.id in markers:
            text += f"<br/><b>detail {markers[node.id]}</b>"
        box = f'[("{text}")]' if node.kind == "data" else f'("{text}")'
        extra = ",expanded" if node.id in markers else ""
        lines.append(f"  {ids[node.id]}{box}:::{node.kind}{extra}")
    for edge in graph.edges:
        arrow = f'-->|"{_text(edge.label)}"|' if edge.label else "-->"
        lines.append(f"  {ids[edge.source]} {arrow} {ids[edge.target]}")
    return "\n".join(lines) + "\n"


def image_names(flows: Flows) -> list[str]:
    """PNG names of the diagrams: ``flow.png``, then ``flow-a.png``, ``flow-b.png``, ...

    Args:
        flows: All diagrams.

    Returns:
        File names, overview first.
    """
    return ["flow.png"] + [f"flow-{LETTERS[i].lower()}.png" for i in range(len(flows.details))]


def flow_sources(flows: Flows) -> list[str]:
    """Mermaid source of every diagram, overview (with detail markers) first.

    Args:
        flows: All diagrams.

    Returns:
        One source per diagram, in :func:`image_names` order.
    """
    markers = {d.node_id: LETTERS[i] for i, d in enumerate(flows.details)}
    return [flow_mermaid(flows.overview, markers), *(flow_mermaid(d.graph) for d in flows.details)]


def render_mermaid(sources: list[str], out: list[Path]) -> list[Path]:
    """Render Mermaid diagrams to PNG with mermaid-cli, all in one call.

    Args:
        sources: Mermaid sources.
        out: One output PNG path per source.

    Returns:
        ``out``.

    Raises:
        RuntimeError: If Node.js (``npx``) is missing or mermaid-cli fails.
    """
    if shutil.which("npx") is None:
        raise RuntimeError("Node.js is not installed: `npx` is needed to render Mermaid diagrams")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "config.json").write_text(json.dumps(MERMAID_THEME))
        blocks = "\n".join(f"```mermaid\n{s}```\n" for s in sources)
        (work / "flows.md").write_text(blocks)
        proc = subprocess.run(
            [*MERMAID_CLI, "--quiet", "-i", "flows.md", "-o", "out.md", "-e", "png",
             "-s", str(SCALE), "-b", "transparent", "-c", "config.json"],
            cwd=work, capture_output=True, text=True, timeout=MERMAID_TIMEOUT_S,
        )  # fmt: skip
        if proc.returncode != 0:
            raise RuntimeError(f"mermaid-cli failed: {proc.stderr.strip()[:300]}")
        for i, path in enumerate(out, start=1):
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(work / f"out-{i}.png", path)
    return out


def render_flows(flows: Flows, out_dir: Path) -> list[str]:
    """Render the overview (with detail markers) and every detail diagram to PNG.

    Args:
        flows: All diagrams.
        out_dir: Output directory.

    Returns:
        File names (:func:`image_names`), overview first.
    """
    names = image_names(flows)
    render_mermaid(flow_sources(flows), [out_dir / n for n in names])
    return names


def legend(graph: FlowGraph) -> list[dict[str, str]]:
    """Legend entries for the node kinds a diagram uses.

    Args:
        graph: Flow diagram.

    Returns:
        ``{"name", "fill", "border"}`` per kind, in a fixed order.
    """
    return [
        {"name": KIND_NAMES[k], "fill": KIND_STYLE[k][0], "border": KIND_STYLE[k][1]}
        for k in KIND_STYLE
        if any(n.kind == k for n in graph.nodes)
    ]
