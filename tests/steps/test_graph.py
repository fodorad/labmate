import pytest

from paper2carousel.schemas import ClaimCard, MethodGraph, Paper
from paper2carousel.steps.graph import (
    KICKERS,
    best_rankdir,
    check_graph,
    graph_dot,
    method_cards,
    plan_graph,
    render_graph,
    render_post_image,
)
from paper2carousel.steps.llm import LLM
from tests.steps.test_visuals import needs_dot

EVIDENCE = (
    "We present a modified multi-modal transformer with linear attention, which considers "
    "RGB texture, iris and eye landmarks and head pose angles. Blink presence and eye state "
    "are estimated for each eye."
)


def graph(**changes):
    data = {
        "nodes": [
            {"id": "rgb", "label": "RGB texture", "kind": "input"},
            {"id": "lmk", "label": "Eye landmarks", "kind": "input"},
            {"id": "tr", "label": "Linear attention transformer", "kind": "component"},
            {"id": "out", "label": "Blink presence", "kind": "output"},
        ],
        "edges": [
            {"source": "rgb", "target": "tr"},
            {"source": "lmk", "target": "tr", "label": "fused"},
            {"source": "tr", "target": "out"},
        ],
        "caption": "Features are fused by a linear attention transformer.",
    }
    data.update(changes)
    return MethodGraph.model_validate(data)


def test_a_grounded_graph_passes():
    assert check_graph(graph(), EVIDENCE, frozenset()) == []


def test_structure_and_grounding_rules():
    bad = graph(
        nodes=[
            {"id": "rgb", "label": "RGB texture", "kind": "input"},
            {"id": "rgb", "label": "Quantum annealer", "kind": "component"},
            {"id": "x", "label": "Accuracy of 99.9 percent on all benchmarks", "kind": "output"},
            {"id": "lonely", "label": "Eye landmarks", "kind": "input"},
        ],
        edges=[{"source": "rgb", "target": "x"}, {"source": "rgb", "target": "ghost"}],
    )
    problems = check_graph(bad, EVIDENCE, frozenset())
    assert "node ids must be unique" in problems
    assert "edge rgb -> ghost uses an unknown node id" in problems
    assert "node lonely has no edge" in problems
    assert "label 'Quantum annealer' names nothing in the evidence" in problems
    assert any("more than 5 words" in p for p in problems)
    assert any("number that is not in the evidence" in p for p in problems)
    # the paper title's words count as grounded
    named = graph(
        nodes=[*graph().nodes[:3], {"id": "out", "label": "BlinkLinMulT", "kind": "output"}]
    )
    assert check_graph(named, EVIDENCE, frozenset({"blinklinmult"})) == []


def test_house_style_dot_is_built_by_code():
    dot = graph_dot(graph(), "TB")
    assert (
        "rankdir=TB" in dot
        and 'fillcolor="#fff8d5"' in dot
        and "Linear attention\\ntransformer" in dot
    )
    assert '"lmk" -> "tr" [label=" fused "];' in dot


@needs_dot
def test_render_graph_and_post_image(tmp_path):
    png = render_graph(graph(), tmp_path / "graph.png")
    assert png.read_bytes().startswith(b"\x89PNG") and best_rankdir(graph()) in ("LR", "TB")
    paper = Paper(paper_id="p", title="BlinkLinMulT", authors=["A", "B"])
    out = render_post_image(graph(), png, paper, "A hook", tmp_path / "post.png")
    import pymupdf

    image = pymupdf.Pixmap(str(out))
    assert (image.width, image.height) == (1080, 1350)
    survey = render_post_image(
        graph(), png, paper, "A hook", tmp_path / "survey.png", kicker=KICKERS["survey"]
    )
    assert survey.read_bytes() != out.read_bytes()


def test_method_cards_come_from_the_task_and_method_blocks():
    cards = [
        ClaimCard(id=f"c{i}", claim="c", evidence_quote="q", kind="method", section="s",
                  page=1, match=100.0)
        for i in range(4)
    ]  # fmt: skip
    picked = method_cards(
        [["c0"], ["c1"], ["c2", "c0"], ["c3"]], ["task", "challenges", "method", "results"], cards
    )
    assert [c.id for c in picked] == ["c0", "c2"]


def test_plan_graph_feeds_rule_violations_back(fake):
    replies = iter(
        [
            graph(nodes=[*graph().nodes[:3], {"id": "out", "label": "Quantum", "kind": "output"}]),
            graph(),
        ]
    )

    def model(body):
        return {"model": body["model"], "message": {"content": next(replies).model_dump_json()}}

    fake.chat_handler = model
    card = ClaimCard(id="c1", claim="c", evidence_quote=EVIDENCE, kind="method", section="s",
                     page=1, match=100.0)  # fmt: skip
    result = plan_graph(
        Paper(paper_id="p", title="T"), ["A bullet."], [card], LLM(fake.client(), "qwen3.6:35b-mlx")
    )
    assert result == graph()
    retry = fake.requests[1][1]["messages"][-1]["content"]
    assert "label 'Quantum' names nothing in the evidence" in retry


def test_graph_needs_enough_nodes():
    with pytest.raises(ValueError):
        graph(nodes=graph().nodes[:2])
