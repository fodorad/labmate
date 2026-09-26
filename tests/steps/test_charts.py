import pymupdf

from paper2carousel.schemas import Bullet, ClaimCard, Claims, Deck, DraftSlide, Paper, SlideText
from paper2carousel.steps.charts import bar_chart_svg, check_chart
from paper2carousel.steps.llm import LLM
from paper2carousel.steps.render import render_deck
from paper2carousel.steps.visuals import choose_visuals
from tests.steps.test_visuals import scripted

EVIDENCE = (
    "On the WMT 2014 English-to-German translation task, the big transformer model "
    "(Transformer (big) in Table 2) outperforms the best previously reported models "
    "(including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art "
    "BLEU score of 28.4. ConvS2S 25.16 GNMT + RL 24.6 Transformer (base model) 27.3 41.0"
)


def test_values_must_come_from_the_evidence_and_keep_their_spelling():
    problems, shown = check_chart(["Transformer (big)", "ConvS2S"], [28.4, "25.16"], EVIDENCE)
    assert problems == [] and shown == ["28.4", "25.16"]
    _, shown = check_chart(["Transformer (base model)", "GNMT + RL"], [41, 24.6], EVIDENCE)
    assert shown == ["41.0", "24.6"]  # as the paper writes it


def test_invented_values_labels_and_bad_shapes_are_rejected():
    problems, _ = check_chart(["Transformer (big)", "ByteNet"], [28.4, 23.75], EVIDENCE)
    assert "value 23.75 for 'ByteNet' is not in the evidence" in problems
    assert "label 'ByteNet' does not name anything in the evidence" in problems
    assert check_chart(["Transformer"], [28.4], EVIDENCE)[0] == ["use 2 to 8 bars, not 1"]
    problems, _ = check_chart(["Transformer", "GNMT"], [28.4], EVIDENCE)
    assert "2 labels but 1 values" in problems
    assert "is not in the evidence" in check_chart(["a1", "GNMT"], ["n/a", 24.6], EVIDENCE)[0][0]


def test_bar_chart_svg_is_deterministic_escaped_and_highlights_the_method():
    svg = bar_chart_svg(
        ["Transformer (big)", "GNMT + RL <x>"], ["28.4", "24.6"], "BLEU", "Transformer (big)"
    )
    assert svg == bar_chart_svg(
        ["Transformer (big)", "GNMT + RL <x>"], ["28.4", "24.6"], "BLEU", "Transformer (big)"
    )
    assert "&lt;x&gt;" in svg and "<x>" not in svg
    assert svg.count('fill-opacity="1"') == 1 and svg.count('fill-opacity="0.45"') == 1
    assert ">BLEU<" in svg and ">28.4<" in svg


def test_agent_draws_a_chart_after_fixing_a_rejected_value(fake, tmp_path):
    claims = Claims(
        cards=[
            ClaimCard(id="c01", claim="c", evidence_quote=EVIDENCE, kind="result",
                      section="Results", page=8, match=100.0)
        ],
        rejected=[],
    )  # fmt: skip
    slide = SlideText(
        title="Big Transformer results",
        bullets=[Bullet(text="28.4 BLEU on WMT 2014 English-to-German.", claim_ids=["c01"])],
    )
    chart = {"labels": ["Transformer (big)", "ConvS2S"], "unit": "BLEU", "caption": "EN-DE."}
    scripted(
        fake,
        ("make_chart", {**chart, "values": [28.4, 26.0]}),
        ("make_chart", {**chart, "values": [28.4, 25.16], "highlight": "Transformer (big)"}),
    )
    llm = LLM(fake.client(), "qwen3.6:35b-mlx")
    result = choose_visuals([slide], [], tmp_path, llm, "", claims)
    assert "value 26.0 for 'ConvS2S' is not in the evidence" in result.steps[0].observation
    visual = result.slides[0]
    assert visual.kind == "chart" and visual.path == "charts/slide1.svg"
    assert (tmp_path / visual.path).read_text().startswith("<svg")
    assert EVIDENCE[:40] in fake.requests[0][1]["messages"][0]["content"]


def test_a_chart_renders_inside_the_carousel(tmp_path):
    (tmp_path / "charts").mkdir()
    svg = bar_chart_svg(
        ["Transformer (big)", "ConvS2S"], ["28.4", "25.16"], "BLEU", "Transformer (big)"
    )
    (tmp_path / "charts" / "slide1.svg").write_text(svg)
    deck = Deck(
        title="t",
        slides=[DraftSlide(title="Results", bullets=["b"], image="charts/slide1.svg")],
    )
    out = render_deck(deck, Paper(paper_id="p", title="P"), tmp_path / "carousel.pdf")
    with pymupdf.open(out) as doc:
        assert "25.16" in doc[1].get_text()
