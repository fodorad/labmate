import pymupdf

from paper2carousel.schemas import Bullet, ClaimCard, Claims, Deck, DraftSlide, Paper, SlideText
from paper2carousel.steps.charts import bar_chart_svg, check_chart
from paper2carousel.steps.llm import LLM
from paper2carousel.steps.render import render_deck
from paper2carousel.steps.visuals import choose_visuals
from tests.steps.test_visuals import scripted

QUOTES = [
    "establishing a new state-of-the-art BLEU score of 28.4",
    "our big model achieves a BLEU score of 41.0, outperforming all of the previously "
    "published single models",
    "ConvS2S 25.16 GNMT + RL 24.6 Transformer (base model) 27.3",
]


def test_values_must_come_from_the_evidence_and_keep_their_spelling():
    problems, shown = check_chart(["Transformer (base)", "ConvS2S"], [27.3, "25.16"], QUOTES)
    assert problems == [] and shown == ["27.3", "25.16"]
    _, shown = check_chart(["Big model", "GNMT + RL"], [41, 24.6], QUOTES)
    assert shown == ["41.0", "24.6"]  # as the paper writes it


def test_a_value_must_be_paired_with_its_own_label_in_one_quote():
    # a real run charted the EN-DE score as "previously published single models"
    problems, _ = check_chart(["Previous single models", "Big model"], [28.4, 41.0], QUOTES)
    assert problems == [
        "no evidence quote gives 28.4 for 'Previous single models'; pair each value with the "
        "name it belongs to in the same quote"
    ]


def test_invented_values_long_labels_and_bad_shapes_are_rejected():
    problems, _ = check_chart(["ConvS2S", "ByteNet"], [25.16, 23.75], QUOTES)
    assert problems == ["value 23.75 for 'ByteNet' is not in the evidence"]
    assert check_chart(["ConvS2S"], [25.16], QUOTES)[0] == ["use 2 to 8 bars, not 1"]
    assert "2 labels but 1 values" in check_chart(["ConvS2S", "GNMT"], [25.16], QUOTES)[0]
    problems, _ = check_chart(["ConvS2S (Gehring et al., 2017)", "GNMT"], [25.16, 24.6], QUOTES)
    assert problems == ["label 'ConvS2S (Gehring et al., 2017)' is longer than 24 characters"]
    assert "is not in the evidence" in check_chart(["x", "GNMT"], ["n/a", 24.6], QUOTES)[0][0]


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
            ClaimCard(id=f"c0{i}", claim="c", evidence_quote=q, kind="result",
                      section="Results", page=8, match=100.0)
            for i, q in enumerate(QUOTES, start=1)
        ],
        rejected=[],
    )  # fmt: skip
    slide = SlideText(
        title="Big Transformer results",
        bullets=[Bullet(text="27.3 BLEU for the base model.", claim_ids=["c01", "c03"])],
    )
    chart = {"labels": ["Transformer (base)", "ConvS2S"], "unit": "BLEU", "caption": "EN-DE."}
    scripted(
        fake,
        ("make_chart", {**chart, "values": [27.3, 26.0]}),
        ("make_chart", {**chart, "values": [27.3, 25.16], "highlight": "Transformer (base)"}),
    )
    llm = LLM(fake.client(), "qwen3.6:35b-mlx")
    result = choose_visuals([slide], [], tmp_path, llm, "", claims)
    assert "value 26.0 for 'ConvS2S' is not in the evidence" in result.steps[0].observation
    visual = result.slides[0]
    assert visual.kind == "chart" and visual.path == "charts/slide1.svg"
    assert (tmp_path / visual.path).read_text().startswith("<svg")
    assert QUOTES[2] in fake.requests[0][1]["messages"][0]["content"]


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
