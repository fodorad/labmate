import pytest

from paper2carousel.schemas import Bullet, Figure, SlideText
from paper2carousel.steps.llm import LLM
from paper2carousel.steps.visuals import choose_visuals, graphviz_available, render_dot

needs_dot = pytest.mark.skipif(not graphviz_available(), reason="Graphviz not installed")

FIGS = [
    Figure(
        id="fig1", number=1, caption="Figure 1: The architecture.", page=3, path="figures/fig1.png"
    )
]
SLIDES = [
    SlideText(title=f"Slide {i}", bullets=[Bullet(text=f"Point {i}.", claim_ids=["c01"])])
    for i in range(1, 3)
]


def scripted(fake, *calls):
    """Answer each chat with the next scripted tool call (or text if the item is a str)."""
    queue = list(calls)

    def handler(body):
        item = queue.pop(0)
        if isinstance(item, str):
            return {"model": body["model"], "message": {"content": item}}
        name, args = item
        return {
            "model": body["model"],
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": args}}],
            },
        }

    fake.chat_handler = handler
    return fake


def agent(fake, tmp_path, slides=SLIDES, figures=FIGS):
    return choose_visuals(slides, figures, tmp_path, LLM(fake.client(), "qwen3.6:35b-mlx"))


def test_figure_then_no_visual(fake, tmp_path):
    scripted(fake, ("use_paper_figure", {"figure_id": "fig1"}), ("no_visual", {"reason": "text"}))
    result = agent(fake, tmp_path)
    assert result.slides[0].source == "fig1" and result.slides[0].caption == "The architecture."
    assert result.slides[1] is None
    assert [s.tool for s in result.steps] == ["use_paper_figure", "no_visual"]
    # the second slide is no longer offered the figure that is already used
    second_prompt = fake.requests[1][1]["messages"][0]["content"]
    assert "fig1 (page 3)" not in second_prompt and "(none)" in second_prompt


def test_errors_are_observed_and_the_agent_can_recover(fake, tmp_path):
    scripted(
        fake,
        ("use_paper_figure", {"figure_id": "fig9"}),
        ("use_paper_figure", {"figure_id": "fig1"}),
        "I think no visual is needed.",
    )
    result = agent(fake, tmp_path)
    assert result.slides == [result.slides[0], None] and result.slides[0].source == "fig1"
    assert "unknown figure id 'fig9'" in result.steps[0].observation
    retry = fake.requests[1][1]["messages"]
    assert retry[-2]["tool_calls"][0]["function"]["name"] == "use_paper_figure"
    assert retry[-1] == {
        "role": "tool",
        "content": result.steps[0].observation,
        "tool_name": "use_paper_figure",
    }
    assert result.steps[-1].tool == "(none)"


def test_budget_exhaustion_means_no_visual(fake, tmp_path):
    scripted(
        fake, *[("use_paper_figure", {"figure_id": "nope"})] * 3, ("no_visual", {"reason": "r"})
    )
    result = agent(fake, tmp_path)
    assert result.slides == [None, None]
    assert [s.slide for s in result.steps] == [1, 1, 1, 2]


def test_used_figure_and_unknown_tool_are_rejected(fake, tmp_path):
    scripted(
        fake,
        ("use_paper_figure", {"figure_id": "fig1"}),
        ("use_paper_figure", {"figure_id": "fig1"}),
        ("draw_unicorn", {}),
        ("no_visual", {"reason": "r"}),
    )
    result = agent(fake, tmp_path)
    assert "already on slide 1" in result.steps[1].observation
    assert "unknown tool 'draw_unicorn'" in result.steps[2].observation


@needs_dot
def test_diagram_errors_are_fed_back_then_fixed(fake, tmp_path):
    scripted(
        fake,
        ("make_diagram", {"dot": "digraph { a -> }", "caption": "broken"}),
        ("make_diagram", {"dot": "digraph { rankdir=LR; a -> b }", "caption": "Flow."}),
        ("no_visual", {"reason": "r"}),
    )
    result = agent(fake, tmp_path, figures=[])
    assert "syntax error" in result.steps[0].observation
    assert result.slides[0].kind == "diagram" and result.slides[0].caption == "Flow."
    assert (tmp_path / result.slides[0].path).read_bytes().startswith(b"\x89PNG")


@needs_dot
def test_render_dot_timeout_and_success(tmp_path, monkeypatch):
    assert render_dot("digraph { x -> y }", tmp_path / "d" / "ok.png") is None
    monkeypatch.setattr("paper2carousel.steps.visuals.DOT_TIMEOUT_S", 0.0001)
    assert "timed out" in render_dot("digraph { x -> y }", tmp_path / "slow.png")


def test_diagram_tool_is_only_offered_with_graphviz(fake, tmp_path, monkeypatch):
    monkeypatch.setattr("paper2carousel.steps.visuals.shutil.which", lambda _: None)
    scripted(fake, ("no_visual", {"reason": "r"}), ("no_visual", {"reason": "r"}))
    agent(fake, tmp_path)
    tools = [t["function"]["name"] for t in fake.requests[0][1]["tools"]]
    assert tools == ["use_paper_figure", "no_visual"]


def test_figure_label_is_stripped_but_not_the_rest_of_the_caption():
    from paper2carousel.steps.visuals import _FIGURE_LABEL

    caption = "Figure 1. BlinkLinMulT: a multimodal transformer."
    assert _FIGURE_LABEL.sub("", caption) == "BlinkLinMulT: a multimodal transformer."
    assert _FIGURE_LABEL.sub("", "Fig. 3: Results.") == "Results."
