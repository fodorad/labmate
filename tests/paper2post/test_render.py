import pymupdf

from labmate.paper2flow.schemas import (
    Bullet,
    FactCheckReport,
    FlowEdge,
    FlowNode,
    FlowOverview,
    Flows,
    Paper,
)
from labmate.paper2flow.steps.flow import render_flows
from labmate.paper2flow.steps.render import diagrams
from labmate.paper2post.render import render_post
from labmate.paper2post.schemas import Link, Post

PAPER = Paper(paper_id="x", title="A Test Paper", url="https://arxiv.org/abs/2401.00001")
POST = Post(
    hook="Linear attention without the accuracy tax",
    takeaways=[
        Bullet(text="Attention costs grow with length.", claim_ids=["c01"]),
        Bullet(text="It reaches 84.6% accuracy.", claim_ids=["c02"]),
    ],
    question="Where would it help you?",
    icons=["target", "chart-bar"],
    links=[Link(label="Paper", url=PAPER.url), Link(label="Website", url="https://adamfodor.com")],
    report=FactCheckReport(rounds=[]),
)


def pipeline(tmp_path):
    flows = Flows(
        overview=FlowOverview(
            title="From tokens to labels",
            nodes=[FlowNode(id="a", label="Input tokens", kind="input"),
                   FlowNode(id="b", label="Linear attention", kind="component"),
                   FlowNode(id="c", label="Class labels", kind="output")],
            edges=[FlowEdge(source="a", target="b"), FlowEdge(source="b", target="c")],
            caption="Tokens pass through linear attention.",
        )
    )  # fmt: skip
    return diagrams(flows, render_flows(flows, tmp_path), tmp_path)[0]


def test_the_post_is_its_text_with_icons_and_links_then_the_pipeline_image(tmp_path):
    out = render_post(POST, PAPER, pipeline(tmp_path), tmp_path / "post.pdf")
    with pymupdf.open(out) as doc:
        assert doc.page_count == 2 and doc[0].rect.width / doc[0].rect.height == 0.8
        text = " ".join(doc[0].get_text().split())
        assert POST.hook in text and "It reaches 84.6% accuracy." in text
        assert "Paper: https://arxiv.org/abs/2401.00001" in text
        assert "Website: https://adamfodor.com" in text
        assert len(doc[0].get_drawings()) > 2  # the two icons are drawn
        assert "END-TO-END DATA FLOW" in doc[1].get_text() and doc[1].get_images()
    icon = (tmp_path / "icons" / "target.svg").read_text()
    assert "currentColor" not in icon and "#ab4c31" in icon  # drawn in the accent colour
