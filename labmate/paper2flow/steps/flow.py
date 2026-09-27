"""Flow diagrams (ORCHESTRATOR–WORKERS): the end-to-end data flow, then its details.

The planner draws the overview, from the raw data to the target output, and picks the
steps that deserve a closer look; one worker call per picked step draws its detail
diagram. The model only proposes graphs as data (typed nodes and edges). Code checks
them — the flow must start at inputs and end at outputs, every label must name something
the paper says, no number may be invented — feeds problems back, and lays the graphs out
with Graphviz, top to bottom, in a fixed house style.
"""

from __future__ import annotations

import html
import subprocess
from collections import Counter
from pathlib import Path

from labmate.core.factcheck import numbers_in
from labmate.core.llm.structured import structured_chat
from labmate.core.model import LLM
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
    PaperType,
)

DOT_TIMEOUT_S = 20
"""Graphviz render timeout."""

DPI = 200
"""Resolution of the diagram PNGs."""

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
    "input": ("#fff8d5", "#c9ad3f", "box"),
    "data": ("#f7f6f4", "#9aa5b1", "cylinder"),
    "component": ("#e3f2f8", "#5f95b0", "box"),
    "process": ("#f1e1dc", "#ab4c31", "box"),
    "output": ("#e4f8d6", "#6fa24c", "box"),
}
"""Fill, border and shape per node kind (the project-page card colours)."""

KIND_NAMES = {
    "input": "Input",
    "data": "Data",
    "component": "Model component",
    "process": "Process",
    "output": "Output",
}
"""Legend entries."""

GOALS: dict[str, str] = {
    "method": "how data flows through the proposed method: from the raw input data, "
    "through preprocessing and the model's components, to the target output",
    "benchmark": "how the benchmark is built and used: from the raw data sources, through "
    "collection, annotation and the tasks, to the evaluation results",
    "survey": "how the survey organises its field: from the problem, through the "
    "categories of approaches and how they are evaluated, to the open challenges",
    "position": "how the argument flows: from the observed situation, through the "
    "evidence and the reasoning, to the proposed position and its consequences",
}
"""What the overview shows, per paper type."""


# --- evidence ------------------------------------------------------------------------------


def flow_cards(outline: Outline, cards: list[ClaimCard]) -> list[ClaimCard]:
    """Claim cards the diagrams may draw on: the task and method blocks, then all method claims.

    Args:
        outline: Approved outline.
        cards: All claim cards.

    Returns:
        Cards in that order, without duplicates.
    """
    by_id = {c.id: c for c in cards}
    ids = [cid for sl in outline.slides if sl.purpose in ("task", "method") for cid in sl.claim_ids]
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
    paper_type: PaperType,
    bullets: list[str],
    cards: list[ClaimCard],
    sections: str,
    llm: LLM,
) -> FlowOverview:
    """Ask the planner for the end-to-end flow and the steps to expand.

    Args:
        paper: The paper.
        paper_type: Route (what the flow shows).
        bullets: The fact-checked method block's bullets.
        cards: Evidence cards (:func:`flow_cards`).
        sections: Evidence text (:func:`flow_sections`).
        llm: Writer model settings.

    Returns:
        An overview that passes :func:`check_overview`.
    """
    evidence = " ".join([*(c.evidence_quote for c in cards), *bullets, sections])
    exempt = frozenset(content_words(paper.title, split_hyphens=True))
    prompt = load_prompt("flow_overview").format(
        title=paper.title,
        goal=GOALS[paper_type],
        bullets="\n".join(f"- {b}" for b in bullets) or "(none)",
        evidence=_evidence(cards, sections),
    )
    return structured_chat(
        llm.backend,
        llm.request(prompt),
        FlowOverview,
        check=lambda g: check_overview(g, evidence, exempt),
    )


def plan_detail(
    paper: Paper,
    overview: FlowOverview,
    node_id: str,
    cards: list[ClaimCard],
    sections: str,
    llm: LLM,
) -> FlowDetail:
    """Ask a worker for the detail diagram of one overview step.

    Args:
        paper: The paper.
        overview: The approved overview.
        node_id: The step to break down.
        cards: Evidence cards.
        sections: Evidence text.
        llm: Writer model settings.

    Returns:
        The detail diagram, passing :func:`check_detail`.
    """
    evidence = " ".join([*(c.evidence_quote for c in cards), sections])
    exempt = frozenset(content_words(paper.title, split_hyphens=True))
    labels = {n.id: n.label for n in overview.nodes}
    incoming = [labels[e.source] for e in overview.edges if e.target == node_id]
    outgoing = [labels[e.target] for e in overview.edges if e.source == node_id]
    prompt = load_prompt("flow_detail").format(
        title=paper.title,
        overview=_describe(overview),
        step=labels[node_id],
        incoming=", ".join(incoming) or "(nothing)",
        outgoing=", ".join(outgoing) or "(nothing)",
        evidence=_evidence(cards, sections),
    )
    graph = structured_chat(
        llm.backend,
        llm.request(prompt),
        FlowGraph,
        check=lambda g: check_detail(g, overview, evidence, exempt),
    )
    return FlowDetail(node_id=node_id, graph=graph)


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


def _wrap(label: str, width: int = 18) -> list[str]:
    lines: list[str] = []
    for word in label.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines or [label]


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', "'") + '"'


def flow_dot(graph: FlowGraph, markers: dict[str, str] | None = None) -> str:
    """Graphviz source in the house style, top to bottom (built by code, never by the model).

    Args:
        graph: Flow diagram.
        markers: Node id to detail letter; those boxes get a "detail A" note and a double
            border, pointing to their detail diagram.

    Returns:
        DOT source.
    """
    markers = markers or {}
    lines = [
        "digraph flow {",
        '  graph [rankdir=TB, bgcolor="transparent", pad="0.3", nodesep="0.45", '
        'ranksep="0.5", splines=true];',
        '  node [style="rounded,filled", fontname="Helvetica", fontsize=15, '
        'fontcolor="#222b35", penwidth=1.6, margin="0.24,0.12"];',
        '  edge [color="#4c6176", penwidth=1.4, arrowsize=0.8, fontname="Helvetica", '
        'fontsize=12, fontcolor="#4c6176"];',
    ]
    for node in graph.nodes:
        fill, border, shape = KIND_STYLE[node.kind]
        text = "<BR/>".join(html.escape(line) for line in _wrap(node.label))
        extra = ""
        if node.id in markers:
            text += (
                f'<BR/><FONT POINT-SIZE="11" COLOR="#ab4c31"><B>detail {markers[node.id]}'
                "</B></FONT>"
            )
            extra = ", peripheries=2"
        lines.append(
            f"  {_quote(node.id)} [label=<{text}>, shape={shape}, "
            f'fillcolor="{fill}", color="{border}"{extra}];'
        )
    for edge in graph.edges:
        attrs = f"label={_quote('  ' + edge.label + '  ')}" if edge.label else ""
        lines.append(f"  {_quote(edge.source)} -> {_quote(edge.target)} [{attrs}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_flow(graph: FlowGraph, out: Path, markers: dict[str, str] | None = None) -> Path:
    """Render a flow diagram to PNG with Graphviz.

    Args:
        graph: Flow diagram.
        out: Output PNG path.
        markers: See :func:`flow_dot`.

    Returns:
        ``out``.

    Raises:
        RuntimeError: If Graphviz fails.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["dot", "-Tpng", f"-Gdpi={DPI}", "-o", str(out.resolve())],
        input=flow_dot(graph, markers),
        capture_output=True,
        text=True,
        timeout=DOT_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Graphviz failed: {proc.stderr.strip()[:300]}")
    return out


def image_names(flows: Flows) -> list[str]:
    """PNG names of the diagrams: ``flow.png``, then ``flow-a.png``, ``flow-b.png``, ...

    Args:
        flows: All diagrams.

    Returns:
        File names, overview first.
    """
    return ["flow.png"] + [f"flow-{LETTERS[i].lower()}.png" for i in range(len(flows.details))]


def render_flows(flows: Flows, out_dir: Path) -> list[str]:
    """Render the overview (with detail markers) and every detail diagram.

    Args:
        flows: All diagrams.
        out_dir: Output directory.

    Returns:
        File names (:func:`image_names`), overview first.
    """
    markers = {d.node_id: LETTERS[i] for i, d in enumerate(flows.details)}
    names = image_names(flows)
    render_flow(flows.overview, out_dir / names[0], markers)
    for detail, name in zip(flows.details, names[1:], strict=True):
        render_flow(detail.graph, out_dir / name)
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
