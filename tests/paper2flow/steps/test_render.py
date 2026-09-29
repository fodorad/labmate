import pymupdf
import pytest

from labmate.paper2flow.schemas import (
    Bullet,
    FactCheckReport,
    FlowDetail,
    FlowEdge,
    FlowGraph,
    FlowNode,
    FlowOverview,
    Flows,
    Paper,
    Post,
    SlideText,
    WrittenSlides,
)
from labmate.paper2flow.steps.flow import render_flows
from labmate.paper2flow.steps.render import (
    Theme,
    author_line,
    blocks,
    diagrams,
    natural_size,
    post_text,
    publication_line,
    render_overview,
    render_post,
)


def flat(page):
    return " ".join(page.get_text().split())


PAPER = Paper(
    paper_id="x",
    title="A Test Paper",
    authors=["Ada Lovelace", "Alan Turing"],
    url="https://arxiv.org/abs/2401.00001",
    date="2024-01-02",
    year=2024,
)
LABELS = ["Task", "Challenges", "Proposed method", "Main results"]


def _node(i, label, kind):
    return FlowNode(id=i, label=label, kind=kind)


def flows() -> Flows:
    overview = FlowOverview(
        title="From tokens to labels",
        nodes=[
            _node("a", "Input tokens", "input"),
            _node("b", "Linear attention", "component"),
            _node("c", "Class labels", "output"),
        ],
        edges=[FlowEdge(source="a", target="b"), FlowEdge(source="b", target="c")],
        caption="Tokens pass through linear attention.",
        expand=["b"],
    )
    detail = FlowGraph(
        title="Inside linear attention",
        nodes=[
            _node("q", "Queries", "data"),
            _node("f", "Feature map", "process"),
            _node("o", "Attention output", "output"),
        ],
        edges=[FlowEdge(source="q", target="f"), FlowEdge(source="f", target="o")],
        caption="A feature map replaces the softmax.",
    )
    return Flows(overview=overview, details=[FlowDetail(node_id="b", graph=detail)])


def slides() -> WrittenSlides:
    return WrittenSlides(
        hook="Linear attention, same accuracy",
        slides=[
            SlideText(
                title=f"About {label}", bullets=[Bullet(text=f"{label} bullet.", claim_ids=["c01"])]
            )
            for label in LABELS
        ],
    )


POST = Post(
    hook="Linear attention without the accuracy tax",
    takeaways=[Bullet(text="It reaches 84.6% accuracy.", claim_ids=["c01"])],
    question="Where would it help you?",
    report=FactCheckReport(rounds=[]),
)


def test_overview_has_paper_blocks_flow_and_details(tmp_path):
    f = flows()
    pages = diagrams(f, render_flows(f, tmp_path), "method", tmp_path)
    out = render_overview(PAPER, blocks(slides(), LABELS), pages, tmp_path / "overview.pdf")
    with pymupdf.open(out) as doc:
        assert doc.page_count == 4
        assert round(doc[0].rect.width) == 595  # A4 portrait
        first = flat(doc[0])
        assert "A Test Paper" in first and "Ada Lovelace, Alan Turing" in first
        assert "arXiv preprint · 2 January 2024" in first
        assert "Proposed method" in flat(doc[1]) and "About Main results" in flat(doc[1])
        assert "END-TO-END DATA FLOW" in flat(doc[2]) and doc[2].get_images()
        assert "DETAIL A · LINEAR ATTENTION" in flat(doc[3])
        assert "1 / 4" in flat(doc[0])
    assert not (tmp_path / "overview.typ").exists()


def test_post_is_the_text_then_one_image_per_diagram(tmp_path):
    f = flows()
    pages = diagrams(f, render_flows(f, tmp_path), "survey", tmp_path)
    with pymupdf.open(render_post(POST, PAPER, pages, tmp_path / "post.pdf")) as doc:
        assert doc.page_count == 3
        assert doc[0].rect.width / doc[0].rect.height == 0.8
        text = flat(doc[0])
        assert "Linear attention without the accuracy tax" in text
        assert "It reaches 84.6% accuracy." in text and "arxiv.org/abs/2401.00001" in text
        assert "HOW THE SURVEY MAPS THE FIELD" in flat(doc[1]) and doc[1].get_images()
        assert "Inside linear attention" in flat(doc[2])


def test_small_diagrams_are_not_blown_up(tmp_path):
    f = flows()
    names = render_flows(f, tmp_path)
    width, height = natural_size(tmp_path / names[1])
    assert 50 < width < 400 and 50 < height < 500  # points at the render resolution


def test_without_a_figure_the_first_page_shows_the_abstract(tmp_path):
    f = flows()
    pages = diagrams(f, render_flows(f, tmp_path), "method", tmp_path)
    paper = PAPER.model_copy(update={"abstract": "We study attention."})
    out = render_overview(paper, blocks(slides(), LABELS), pages, tmp_path / "o.pdf")
    with pymupdf.open(out) as doc:
        assert "We study attention." in flat(doc[0])


def test_paper_text_cannot_inject_typst_markup(tmp_path):
    nasty = '#set page(fill: red) *bold* $x$ @ref <label> [x] // comment "quote"'
    f = flows()
    pages = diagrams(f, render_flows(f, tmp_path), "method", tmp_path)
    paper = PAPER.model_copy(update={"title": nasty})
    out = render_overview(paper, blocks(slides(), LABELS), pages, tmp_path / "o.pdf", theme=Theme())
    with pymupdf.open(out) as doc:
        assert "#set page(fill: red)" in flat(doc[0])


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({}, "arXiv preprint · 2 January 2024"),
        ({"venue": "NeurIPS 2024", "date": ""}, "NeurIPS 2024"),
        (
            {"venue": "J. Imaging 2023, 9, 196", "date": "21 September 2023"},
            "J. Imaging 2023, 9, 196 · 21 September 2023",
        ),
        ({"url": "", "date": "", "year": 2019}, "2019"),
        ({"url": "", "date": "", "year": None}, ""),
    ],
)
def test_publication_line(update, expected):
    assert publication_line(PAPER.model_copy(update=update)) == expected


def test_author_line_and_post_text():
    assert author_line(["A", "B", "C", "D", "E"]) == "A, B, C, D et al."
    assert post_text(POST, PAPER) == [
        POST.hook,
        "It reaches 84.6% accuracy.",
        POST.question,
        "Paper: A Test Paper https://arxiv.org/abs/2401.00001",
    ]
