import json
import shutil

import pytest

from labmate.core.model import LLM
from labmate.paper2flow.schemas import (
    ClaimCard,
    FlowEdge,
    FlowGraph,
    FlowNode,
    FlowOverview,
    Outline,
    OutlineSlide,
    Paper,
    Section,
)
from labmate.paper2flow.steps.flow import (
    assemble_flows,
    check_detail,
    check_graph,
    check_overview,
    detail_order,
    flow_cards,
    flow_dot,
    flow_sections,
    image_names,
    legend,
    plan_detail,
    plan_overview,
    render_flow,
)

needs_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz not installed")

EVIDENCE = (
    "We present a modified multi-modal transformer with linear attention, which considers "
    "RGB texture, iris and eye landmarks and head pose angles. Cross-modal transformers "
    "translate landmark features into the RGB embedding space. Blink presence and eye state "
    "are estimated for each eye from 15 frames."
)
EXEMPT = frozenset({"blinklinmult"})


def node(i, label, kind):
    return FlowNode(id=i, label=label, kind=kind)


def edge(a, b, label=""):
    return FlowEdge(source=a, target=b, label=label)


def overview(**changes) -> FlowOverview:
    base = FlowOverview(
        title="From video to blinks",
        nodes=[
            node("rgb", "RGB texture", "input"),
            node("lm", "Eye landmarks", "input"),
            node("x", "Cross-modal transformer", "component"),
            node("out", "Blink presence", "output"),
        ],
        edges=[edge("rgb", "x"), edge("lm", "x", "landmark features"), edge("x", "out")],
        caption="Inputs are fused and blinks are estimated.",
        expand=["x"],
    )
    return base.model_copy(update=changes)


def detail(**changes) -> FlowGraph:
    base = FlowGraph(
        title="Inside the cross-modal transformer",
        nodes=[
            node("lf", "Landmark features", "data"),
            node("tr", "Translate features", "process"),
            node("emb", "RGB embedding space", "data"),
        ],
        edges=[edge("lf", "tr"), edge("tr", "emb")],
        caption="Landmark features are translated into the RGB embedding space.",
    )
    return base.model_copy(update=changes)


def test_a_valid_overview_and_detail_pass():
    assert check_overview(overview(), EVIDENCE, EXEMPT) == []
    assert check_detail(detail(), overview(), EVIDENCE, EXEMPT) == []


def test_flow_must_start_at_inputs_and_end_at_outputs():
    g = overview(nodes=[node("rgb", "RGB texture", "component"), *overview().nodes[1:3],
                        node("out", "Blink presence", "process")])  # fmt: skip
    problems = check_graph(g, EVIDENCE, EXEMPT)
    assert any("starts at 'RGB texture', a component" in p for p in problems)
    assert any("ends at 'Blink presence', a process" in p for p in problems)
    # a detail may end in data (what the step produces)
    assert check_graph(detail(), EVIDENCE, EXEMPT, sink_kinds=("output", "data")) == []


def test_structure_rules():
    g = overview(
        nodes=[*overview().nodes, node("x", "Head pose angles", "input")],
        edges=[*overview().edges, edge("x", "ghost"), edge("out", "out"),
               edge("rgb", "x", "far too many words on this")],
    )  # fmt: skip
    problems = check_graph(g, EVIDENCE, EXEMPT, max_nodes=4)
    assert "node ids must be unique" in problems
    assert "use at most 4 nodes, not 5" in problems
    assert "edge x -> ghost uses an unknown node id" in problems
    assert "edge out -> out points at itself" in problems
    assert any("has more than 4 words" in p for p in problems)
    lonely = overview(nodes=[*overview().nodes, node("hp", "Head pose angles", "input")])
    assert "node hp has no edge" in check_graph(lonely, EVIDENCE, EXEMPT)


def test_labels_must_be_grounded_short_and_number_free():
    long_label = "Blink presence from 30 frames per eye"
    g = overview(nodes=[node("rgb", "Quantum spectrogram", "input"), *overview().nodes[1:3],
                        node("out", long_label, "output")])  # fmt: skip
    problems = check_graph(g, EVIDENCE, EXEMPT)
    assert "label 'Quantum spectrogram' names nothing in the evidence" in problems
    assert any("more than 5 words" in p for p in problems)
    assert any("has a number that is not in the evidence" in p for p in problems)
    # the paper title's words need no evidence
    named = overview(nodes=[node("rgb", "BlinkLinMulT input", "input"), *overview().nodes[1:]])
    assert check_graph(named, EVIDENCE, EXEMPT) == []


def test_overview_expand_rules():
    assert "expand names unknown node id 'nope'" in check_overview(
        overview(expand=["nope"]), EVIDENCE, EXEMPT
    )
    assert any("only a component or process" in p for p in
               check_overview(overview(expand=["rgb"]), EVIDENCE, EXEMPT))  # fmt: skip
    assert "expand lists a node twice" in check_overview(
        overview(expand=["x", "x"]), EVIDENCE, EXEMPT
    )
    assert any(
        "expand 1 to 3 steps" in p for p in check_overview(overview(expand=[]), EVIDENCE, EXEMPT)
    )


def test_detail_must_add_new_boxes():
    copy = detail(nodes=[node("a", "RGB texture", "data"), node("b", "Cross-modal transformer",
                  "process"), node("c", "Blink presence", "output")])  # fmt: skip
    problems = check_detail(copy, overview(), EVIDENCE, EXEMPT)
    assert any("3 of 3 boxes repeat the overview" in p for p in problems)


def test_detail_order_and_assembly():
    ov = overview(
        nodes=[*overview().nodes[:2], node("p", "Head pose", "process"), *overview().nodes[2:]],
        expand=["x", "p"],
    )
    assert detail_order(ov) == ["p", "x"]  # in overview order, not expand order
    from labmate.paper2flow.schemas import FlowDetail

    parts = [FlowDetail(node_id="x", graph=detail()), FlowDetail(node_id="p", graph=detail())]
    flows = assemble_flows(ov, parts)
    assert [d.node_id for d in flows.details] == ["p", "x"]
    assert image_names(flows) == ["flow.png", "flow-a.png", "flow-b.png"]


def test_dot_is_top_to_bottom_escaped_and_marks_expanded_steps():
    g = overview(nodes=[node("rgb", 'RGB "texture" & <more>', "input"), *overview().nodes[1:]])
    dot = flow_dot(g, {"x": "A"})
    assert "rankdir=TB" in dot
    assert "RGB &quot;texture&quot; &amp;<BR/>&lt;more&gt;" in dot  # escaped, wrapped
    assert "detail A" in dot and "peripheries=2" in dot
    assert 'label="  landmark features  "' in dot
    assert [e["name"] for e in legend(g)] == ["Input", "Model component", "Output"]


@needs_dot
def test_render_flow_writes_a_png(tmp_path):
    out = render_flow(overview(), tmp_path / "f.png", {"x": "A"})
    assert out.read_bytes().startswith(b"\x89PNG")


@needs_dot
def test_render_flow_reports_graphviz_errors(tmp_path, monkeypatch):
    import labmate.paper2flow.steps.flow as flow

    monkeypatch.setattr(flow, "flow_dot", lambda g, m=None: "digraph { a -> }")
    with pytest.raises(RuntimeError, match="Graphviz failed"):
        render_flow(overview(), tmp_path / "f.png")


CARDS = [
    ClaimCard(id="c01", claim="c", evidence_quote="RGB texture", kind="task", section="Intro",
              page=1, match=100),
    ClaimCard(id="c02", claim="c", evidence_quote="linear attention", kind="method",
              section="Method", page=2, match=100),
    ClaimCard(id="c03", claim="c", evidence_quote="0.99 F1", kind="result", section="Results",
              page=3, match=100),
    ClaimCard(id="c04", claim="c", evidence_quote="landmarks", kind="contribution",
              section="Method", page=2, match=100),
]  # fmt: skip
OUTLINE = Outline(
    hook="h",
    slides=[
        OutlineSlide(title="t", purpose="task", claim_ids=["c01"]),
        OutlineSlide(title="c", purpose="challenges", claim_ids=["c03"]),
        OutlineSlide(title="m", purpose="method", claim_ids=["c02"]),
        OutlineSlide(title="r", purpose="results", claim_ids=["c03"]),
    ],
)
PAPER = Paper(
    paper_id="x",
    title="BlinkLinMulT",
    sections=[
        Section(title="Intro", page=1, text="intro words " * 5),
        Section(title="Method", page=2, text=EVIDENCE),
        Section(title="Results", page=3, text="results " * 10),
    ],
)


def test_evidence_cards_and_sections():
    cards = flow_cards(OUTLINE, CARDS)
    assert [c.id for c in cards] == ["c01", "c02", "c04"]  # task + method blocks, then methods
    text = flow_sections(PAPER, cards)
    assert text.startswith("## Method\n")  # most-cited section first
    assert "## Intro" in text and "## Results" not in text
    assert len(flow_sections(PAPER, cards, max_words=5).split()) == 2 + 5  # "## Method" + 5


def test_planner_and_worker_prompts_and_checks(fake):
    replies = iter([overview(expand=["rgb"]), overview(), detail()])
    seen = []

    def handler(body):
        seen.append(body["messages"][-1]["content"])
        return {"model": body["model"], "message": {"content": next(replies).model_dump_json()}}

    fake.chat_handler = handler
    llm = LLM(fake.client(), "qwen3.6:35b-mlx")
    cards = flow_cards(OUTLINE, CARDS)
    sections = flow_sections(PAPER, cards)
    ov = plan_overview(PAPER, "method", ["It fuses RGB and landmarks."], cards, sections, llm)
    assert ov == overview()
    assert "from the raw input data" in seen[0] and '- "linear attention"' in seen[0]
    assert "only a component or process" in seen[1]  # the rule was fed back
    d = plan_detail(PAPER, ov, "x", cards, sections, llm)
    assert d.node_id == "x" and d.graph == detail()
    assert 'inside the step "Cross-modal transformer"' in seen[2]
    assert "It receives: RGB texture, Eye landmarks. It feeds: Blink presence." in seen[2]
    assert json.loads(ov.model_dump_json())["expand"] == ["x"]
