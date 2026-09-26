"""The LinkedIn post image: the proposed method as a pipeline graph.

The writer model proposes the graph as data (typed nodes and edges); code checks that
every label names something the evidence names, then lays it out with Graphviz in a
fixed house style, so the model decides *what* is on the image and code decides *how*
it looks. Typst puts the graph on a 4:5 image with the hook and the paper's title.
"""

from __future__ import annotations

import json
import math
import subprocess
from importlib.resources import files
from pathlib import Path

import typst

from paper2carousel.llm.structured import structured_chat
from paper2carousel.schemas import ClaimCard, MethodGraph, Paper
from paper2carousel.steps.factcheck import numbers_in
from paper2carousel.steps.llm import LLM, load_prompt
from paper2carousel.steps.render import FONTS_DIR, Theme, author_line
from paper2carousel.steps.visuals import DOT_TIMEOUT_S, content_words

MAX_LABEL_WORDS = 5
"""Longest node label."""

GRAPH_ASPECT = 1.35
"""Width / height of the graph area on the post image."""

KIND_STYLE = {
    "input": ("#fff8d5", "#c9ad3f", "box"),
    "data": ("#f7f6f4", "#9aa5b1", "cylinder"),
    "component": ("#e3f2f8", "#5f95b0", "box"),
    "process": ("#f1e1dc", "#ab4c31", "box"),
    "output": ("#e4f8d6", "#6fa24c", "box"),
}
"""Fill, border and shape per node kind (the project-page card colours)."""

KICKERS = {
    "method": "Proposed method",
    "benchmark": "How the benchmark works",
    "survey": "How the survey maps the field",
    "position": "The argument",
}
"""Small heading above the hook, per paper type (a survey has no "proposed method")."""

KIND_NAMES = {
    "input": "Input",
    "data": "Data",
    "component": "Model component",
    "process": "Process",
    "output": "Output",
}
"""Legend entries."""


def check_graph(graph: MethodGraph, evidence: str, exempt: frozenset[str]) -> list[str]:
    """Structural rules plus grounding: every label must name something in the evidence.

    Args:
        graph: Proposed graph.
        evidence: Evidence quotes and slide text the graph may draw on.
        exempt: Words that need no evidence (the paper title's).

    Returns:
        Problems; empty if the graph is valid.
    """
    problems = []
    ids = [n.id for n in graph.nodes]
    if len(set(ids)) != len(ids):
        problems.append("node ids must be unique")
    known = set(ids)
    for e in graph.edges:
        if e.source not in known or e.target not in known:
            problems.append(f"edge {e.source} -> {e.target} uses an unknown node id")
    linked = {e.source for e in graph.edges} | {e.target for e in graph.edges}
    problems += [f"node {n} has no edge" for n in ids if n not in linked]
    words = content_words(evidence, split_hyphens=True) | exempt
    numbers = numbers_in(evidence)
    for node in graph.nodes:
        if len(node.label.split()) > MAX_LABEL_WORDS:
            problems.append(f"label {node.label!r} has more than {MAX_LABEL_WORDS} words")
        if not content_words(node.label, split_hyphens=True) & words:
            problems.append(f"label {node.label!r} names nothing in the evidence")
        if numbers_in(node.label) - numbers:
            problems.append(f"label {node.label!r} has a number that is not in the evidence")
    return problems


def plan_graph(paper: Paper, bullets: list[str], cards: list[ClaimCard], llm: LLM) -> MethodGraph:
    """Ask the writer for the method graph, with the rules fed back on violation.

    Args:
        paper: The paper.
        bullets: The (fact-checked) method slide's bullets.
        cards: Claim cards of the task and method blocks (their quotes are the evidence).
        llm: Writer model settings.

    Returns:
        A graph that passes :func:`check_graph`.
    """
    quotes = [c.evidence_quote for c in cards]
    evidence = " ".join([*quotes, *bullets])
    exempt = frozenset(content_words(paper.title, split_hyphens=True))
    prompt = load_prompt("graph").format(
        title=paper.title,
        bullets="\n".join(f"- {b}" for b in bullets) or "(none)",
        evidence="\n".join(f'- "{q}"' for q in quotes) or "(none)",
    )
    return structured_chat(
        llm.backend,
        llm.request(prompt),
        MethodGraph,
        check=lambda g: check_graph(g, evidence, exempt),
    )


def _quote(text: str) -> str:
    return '"' + text.replace('"', "'") + '"'


def graph_dot(graph: MethodGraph, rankdir: str = "LR") -> str:
    """Graphviz source in the house style (built by code, never by the model).

    Args:
        graph: Method graph.
        rankdir: ``LR`` (left to right) or ``TB`` (top to bottom).

    Returns:
        DOT source.
    """
    lines = [
        "digraph method {",
        f'  graph [rankdir={rankdir}, bgcolor="transparent", pad="0.25", nodesep="0.35", '
        'ranksep="0.55", splines=true];',
        '  node [style="rounded,filled", fontname="Helvetica", fontsize=15, '
        'fontcolor="#222b35", penwidth=1.6, margin="0.22,0.12"];',
        '  edge [color="#4c6176", penwidth=1.4, arrowsize=0.8, fontname="Helvetica", '
        'fontsize=11, fontcolor="#4c6176"];',
    ]
    for node in graph.nodes:
        fill, border, shape = KIND_STYLE[node.kind]
        label = "\\n".join(_wrap(node.label))
        lines.append(
            f"  {_quote(node.id)} [label={_quote(label)}, shape={shape}, "
            f'fillcolor="{fill}", color="{border}"];'
        )
    for edge in graph.edges:
        label = f", label={_quote(' ' + edge.label + ' ')}" if edge.label else ""
        lines.append(f"  {_quote(edge.source)} -> {_quote(edge.target)} [{label.lstrip(', ')}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _wrap(label: str, width: int = 16) -> list[str]:
    lines: list[str] = []
    for word in label.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines or [label]


def _aspect(dot: str) -> float:
    """Width / height of Graphviz's layout of ``dot``."""
    plain = subprocess.run(
        ["dot", "-Tplain"], input=dot, capture_output=True, text=True, timeout=DOT_TIMEOUT_S
    ).stdout.split()
    return float(plain[2]) / float(plain[3]) if len(plain) > 3 and float(plain[3]) else 1.0


def best_rankdir(graph: MethodGraph, target_aspect: float = GRAPH_ASPECT) -> str:
    """The orientation whose layout best fills the image's graph area.

    Args:
        graph: Method graph.
        target_aspect: Width / height of the area the graph is shown in.

    Returns:
        ``LR`` or ``TB``.
    """
    scores = {
        rankdir: abs(math.log(_aspect(graph_dot(graph, rankdir)) / target_aspect))
        for rankdir in ("LR", "TB")
    }
    return min(scores, key=lambda r: scores[r])


def render_graph(graph: MethodGraph, out: Path) -> Path:
    """Render the graph to PNG with Graphviz, in the orientation that fills the image best.

    Args:
        graph: Method graph.
        out: Output PNG path.

    Returns:
        ``out``.

    Raises:
        RuntimeError: If Graphviz fails.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["dot", "-Tpng", "-Gdpi=220", "-o", str(out.resolve())],
        input=graph_dot(graph, best_rankdir(graph)),
        capture_output=True,
        text=True,
        timeout=DOT_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Graphviz failed: {proc.stderr.strip()[:300]}")
    return out


def render_post_image(
    graph: MethodGraph,
    graph_png: Path,
    paper: Paper,
    hook: str,
    out: Path,
    theme: Theme | None = None,
    ppi: int = 144,
    kicker: str = KICKERS["method"],
) -> Path:
    """Compose the post image: hook, paper, the method graph and a legend (4:5, PNG).

    Args:
        graph: Method graph (for the caption and legend).
        graph_png: Rendered graph, in the same directory as ``out``.
        paper: The paper.
        hook: Headline (the post's first line).
        out: Output PNG path (1080 x 1350 at the default ``ppi``).
        theme: Colours.
        ppi: Pixels per inch (144 gives 1080 px width for the 540 pt page).
        kicker: Small heading above the hook (see :data:`KICKERS`).

    Returns:
        ``out``.
    """
    kinds = [k for k in KIND_STYLE if any(n.kind == k for n in graph.nodes)]
    data = {
        "hook": hook,
        "kicker": kicker,
        "title": paper.title,
        "authors": author_line(paper.authors),
        "graph": graph_png.name,
        "caption": graph.caption,
        "legend": [
            {"name": KIND_NAMES[k], "fill": KIND_STYLE[k][0], "border": KIND_STYLE[k][1]}
            for k in kinds
        ],
        "theme": (theme or Theme()).model_dump(),
    }
    source = out.with_suffix(".typ")
    source.write_text(files("paper2carousel.templates").joinpath("post.typ").read_text())
    png = typst.compile(
        str(source),
        root=str(out.parent),
        font_paths=[str(FONTS_DIR)],
        sys_inputs={"post": json.dumps(data)},
        format="png",
        ppi=ppi,
    )
    out.write_bytes(png if isinstance(png, bytes) else png[0])
    return out


def method_cards(
    slide_claims: list[list[str]], purposes: list[str], cards: list[ClaimCard]
) -> list[ClaimCard]:
    """Claim cards of the task and method blocks, in order, without duplicates.

    Args:
        slide_claims: Claim ids per outline slide.
        purposes: Purpose per outline slide.
        cards: All claim cards.

    Returns:
        The cards the graph may draw on.
    """
    by_id = {c.id: c for c in cards}
    ids = [
        cid
        for ids, purpose in zip(slide_claims, purposes, strict=True)
        if purpose in ("task", "method")
        for cid in ids
    ]
    return [by_id[i] for i in dict.fromkeys(ids) if i in by_id]
