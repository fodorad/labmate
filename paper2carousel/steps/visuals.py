"""Step 7 (AGENT): pick a visual for each slide by calling tools, in a bounded loop.

This is the one step where the path genuinely differs per slide, so the model chooses:
reuse a figure from the paper, draw a Graphviz diagram, or no visual. Tool errors (an
unknown figure id, a figure already used, invalid DOT) come back as observations, and
the model can correct itself, within ``MAX_STEPS`` tool calls per slide. Slides are
handled in order, so a figure is never used twice.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper2carousel.llm.types import Message, ToolCall
from paper2carousel.schemas import AgentStep, Figure, SlideText, Visual, Visuals
from paper2carousel.steps.llm import LLM, load_prompt

MAX_STEPS = 3
"""Tool calls allowed per slide before the agent is stopped (no visual)."""

DOT_TIMEOUT_S = 20
"""Graphviz render timeout."""


_FIGURE_LABEL = re.compile(r"^\s*(Figure|Fig\.)\s*\d+\s*[.:]\s*", re.IGNORECASE)
"""The "Figure 3." / "Fig. 3:" prefix, stripped from captions shown on slides."""


def _tool(name: str, description: str, **params: tuple[str, str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": t, "description": d} for k, (t, d) in params.items()},
                "required": list(params),
            },
        },
    }


USE_FIGURE = _tool(
    "use_paper_figure",
    "Put one of the paper's figures on this slide.",
    figure_id=("string", "Figure id, e.g. 'fig2'."),
)
MAKE_DIAGRAM = _tool(
    "make_diagram",
    "Draw a small diagram with Graphviz. Use rankdir=LR, at most 8 nodes and short "
    "labels; colours and fonts are applied automatically.",
    dot=("string", "Complete Graphviz DOT source, e.g. 'digraph { rankdir=LR; a -> b }'."),
    caption=("string", "One-line caption for the diagram."),
)
NO_VISUAL = _tool(
    "no_visual", "Leave this slide without a visual.", reason=("string", "Short reason.")
)


def graphviz_available() -> bool:
    """Whether the ``dot`` binary is on PATH (the diagram tool is only offered if so).

    Returns:
        True if Graphviz is installed.
    """
    return shutil.which("dot") is not None


DOT_STYLE = (
    'graph [bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"]; '
    'node [shape=box, style="rounded,filled", fillcolor="#f1e1dc", color="#ab4c31", '
    'fontname="Helvetica", fontcolor="#222b35", fontsize=13, margin="0.18,0.08"]; '
    'edge [color="#4c6176", arrowsize=0.7]; '
)
"""House style prepended to every diagram; the model's own attributes still override it."""


def style_dot(dot: str) -> str:
    """Insert the house style right after the graph's opening brace.

    Args:
        dot: Graphviz source from the model.

    Returns:
        Styled source (unchanged if there is no opening brace).
    """
    brace = dot.find("{")
    return dot if brace < 0 else dot[: brace + 1] + " " + DOT_STYLE + dot[brace + 1 :]


def render_dot(dot: str, out: Path) -> str | None:
    """Render DOT source to PNG.

    Args:
        dot: Graphviz source.
        out: Output PNG path.

    Returns:
        ``None`` on success, otherwise the error message (fed back to the agent).
    """
    out = out.resolve()  # dot runs with cwd=out.parent, so a relative -o path would nest
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["dot", "-Tpng", "-Gdpi=200", "-o", str(out)],
            input=style_dot(dot),
            capture_output=True,
            text=True,
            timeout=DOT_TIMEOUT_S,
            cwd=out.parent,
        )
    except subprocess.TimeoutExpired:
        return f"Graphviz timed out after {DOT_TIMEOUT_S}s; simplify the diagram"
    if proc.returncode != 0 or not out.exists():
        return proc.stderr.strip()[:400] or "Graphviz failed without a message"
    return None


_STOPWORDS = frozenset(
    "about above after also among another based because been being between both could does "
    "each from have into its more most other over same show shows shown such than that the "
    "their them then there these they this those through under using used uses very were "
    "what when where which while with within without would case figure".split()
)


def slide_caption(caption: str) -> str:
    """Caption as shown on a slide: without the "Figure N." label, first sentence only.

    Args:
        caption: Caption from the paper.

    Returns:
        A short caption.
    """
    text = _FIGURE_LABEL.sub("", caption).strip() or caption
    first = re.split(r"(?<=[a-z0-9)\]])\.\s+(?=[A-Z])", text, maxsplit=1)[0]
    return first if first.endswith(".") else first + "."


def content_words(
    text: str, ignore: frozenset[str] = frozenset(), split_hyphens: bool = False
) -> set[str]:
    """Topic words of a text: lower-cased words of 4+ letters, crude plural stripping.

    Hyphenated words are joined ("multi-modal" matches "multimodal"), so a compound counts
    once. With ``split_hyphens`` their parts are added too, which is how the paper title's
    words are collected ("Transformer-Based" yields "transformer").

    Args:
        text: Any text.
        ignore: Words to leave out (e.g. the paper title's words, which every slide shares).
        split_hyphens: Also include the parts of hyphenated words.

    Returns:
        The set of content words.
    """
    text = text.lower()
    raw = re.findall(r"[a-z]{4,}", re.sub(r"(?<=[a-z])-(?=[a-z])", "", text))
    if split_hyphens:
        raw += re.findall(r"[a-z]{4,}", text)
    words = {w[:-1] if len(w) > 4 and w.endswith("s") else w for w in raw}
    return words - _STOPWORDS - ignore


@dataclass
class _Toolbox:
    """Executes the agent's tool calls for one run.

    ``use_paper_figure`` refuses a figure whose caption shares no topic word with the slide
    (ignoring the paper title's words): a cheap check against placing, say, a head-pose
    plot on an architecture slide. The refusal is returned to the model as an observation.
    """

    figures: dict[str, Figure]
    out_dir: Path
    title_words: frozenset[str] = frozenset()
    used: dict[str, int] = field(default_factory=dict)

    def run(
        self, slide: int, call: ToolCall, slide_text: str = ""
    ) -> tuple[str, Visual | None, bool]:
        """Execute a call. Returns (observation, visual, done)."""
        name, args = call.function.name, call.function.arguments
        if name == "use_paper_figure":
            fid = str(args.get("figure_id", ""))
            if fid not in self.figures:
                return (
                    f"error: unknown figure id {fid!r}; available: {sorted(self.figures)}",
                    None,
                    False,
                )
            if fid in self.used:
                return f"error: {fid} is already on slide {self.used[fid]}", None, False
            fig = self.figures[fid]
            caption = slide_caption(fig.caption)
            shared = content_words(caption, self.title_words) & content_words(
                slide_text, self.title_words
            )
            if not shared:
                return (
                    f"error: {fid} ({caption[:80]}) shows a different topic than this slide; "
                    "choose a figure about this slide's content, make_diagram, or no_visual",
                    None,
                    False,
                )
            self.used[fid] = slide
            return (
                f"ok: {fid} placed",
                Visual(kind="figure", source=fid, path=fig.path, caption=caption),
                True,
            )
        if name == "make_diagram":
            path = self.out_dir / "diagrams" / f"slide{slide}.png"
            error = render_dot(str(args.get("dot", "")), path)
            if error:
                return f"error: Graphviz could not render the DOT source: {error}", None, False
            caption = str(args.get("caption", "")).strip()
            rel = str(path.relative_to(self.out_dir))
            return (
                "ok: diagram rendered",
                Visual(kind="diagram", source="diagram", path=rel, caption=caption),
                True,
            )
        if name == "no_visual":
            return "ok: no visual", None, True
        return f"error: unknown tool {name!r}", None, False


def _format_figures(figures: list[Figure]) -> str:
    return "\n".join(f"{f.id} (page {f.page}): {f.caption}" for f in figures) or "(none)"


def _slide_text(slide: SlideText) -> str:
    return " ".join([slide.title, *(b.text for b in slide.bullets)])


def best_slides(
    slides: list[SlideText], figures: list[Figure], title_words: frozenset[str] = frozenset()
) -> dict[str, set[int]]:
    """For each figure, the slide(s) whose text shares the most topic words with its caption.

    The agent is only offered a figure on its best-matching slide(s); a greedy pass in
    slide order would otherwise spend the architecture diagram on the problem slide.

    Args:
        slides: Final slides.
        figures: Paper figures.
        title_words: Words to ignore (the paper title's).

    Returns:
        Figure id to 1-based slide numbers (empty if no slide shares a topic word).
    """
    words = [content_words(_slide_text(s), title_words) for s in slides]
    result: dict[str, set[int]] = {}
    for fig in figures:
        caption = content_words(slide_caption(fig.caption), title_words)
        scores = [len(caption & w) for w in words]
        top = max(scores, default=0)
        result[fig.id] = {i for i, sc in enumerate(scores, start=1) if top and sc == top}
    return result


def choose_visuals(
    slides: list[SlideText],
    figures: list[Figure],
    out_dir: Path,
    llm: LLM,
    paper_title: str = "",
) -> Visuals:
    """Run the visuals agent over all slides, in order.

    Args:
        slides: Final (fact-checked) slides.
        figures: Figures cropped from the paper.
        out_dir: The paper's run directory (figure paths are relative to it; diagrams go
            to ``out_dir/diagrams``).
        llm: Model settings (the writer model; tool calling was verified by the probe).
        paper_title: Its words are ignored when matching figure captions to slides.

    Returns:
        One optional visual per slide and the complete tool-call log.
    """
    tools = [USE_FIGURE, NO_VISUAL]
    if graphviz_available():
        tools.insert(1, MAKE_DIAGRAM)
    title_words = frozenset(content_words(paper_title, split_hyphens=True))
    box = _Toolbox({f.id: f for f in figures}, out_dir, title_words)
    carousel = "\n".join(f"{i}. {s.title}" for i, s in enumerate(slides, start=1))
    offered = best_slides(slides, figures, title_words)
    template = load_prompt("visuals")
    result = Visuals(slides=[])

    for position, slide in enumerate(slides, start=1):
        prompt = template.format(
            position=position,
            total=len(slides),
            title=slide.title,
            bullets="\n".join(f"- {b.text}" for b in slide.bullets),
            carousel=carousel,
            figures=_format_figures(
                [f for f in figures if f.id not in box.used and position in offered[f.id]]
            ),
        )
        messages = [Message(role="user", content=prompt)]
        visual: Visual | None = None
        for _ in range(MAX_STEPS):
            request = llm.request(prompt).model_copy(update={"messages": messages, "tools": tools})
            response = llm.backend.chat(request)
            if not response.tool_calls:
                result.steps.append(
                    AgentStep(
                        slide=position, tool="(none)", arguments={}, observation="no tool call"
                    )
                )
                break
            call = response.tool_calls[0]
            observation, visual, done = box.run(position, call, _slide_text(slide))
            result.steps.append(
                AgentStep(
                    slide=position,
                    tool=call.function.name,
                    arguments=call.function.arguments,
                    observation=observation,
                )
            )
            if done:
                break
            messages = [
                *messages,
                Message(role="assistant", content=response.content, tool_calls=[call]),
                Message(role="tool", content=observation, tool_name=call.function.name),
            ]
        result.slides.append(visual)
    return result
