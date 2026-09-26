import pymupdf

from paper2carousel.schemas import Deck, DraftSlide, Paper
from paper2carousel.steps.render import Theme, render_deck
from tests.conftest import DECK


def flat(page):
    return " ".join(page.get_text().split())


PAPER = Paper(paper_id="x", title="A Test Paper", url="https://arxiv.org/abs/2401.00001")


def test_renders_cover_plus_one_page_per_slide_at_4_by_5(tmp_path):
    out = render_deck(DECK, PAPER, tmp_path / "out" / "c.pdf")
    with pymupdf.open(out) as doc:
        assert doc.page_count == len(DECK.slides) + 1
        assert doc[0].rect.width / doc[0].rect.height == 0.8
        assert "Linear attention, same accuracy" in flat(doc[0])
        assert "84.6% accuracy" in flat(doc[3])
        assert "arxiv.org/abs/2401.00001" in flat(doc[0])
        assert "adamfodor.com" in flat(doc[1]) and "2 / 4" in flat(doc[1])


def test_paper_text_cannot_inject_typst_markup(tmp_path):
    nasty = '#set page(fill: red) *bold* $x$ @ref <label> [x] // comment "quote"'
    deck = Deck(title=nasty, slides=[DraftSlide(title=nasty, bullets=[nasty])] * 3)
    with pymupdf.open(render_deck(deck, PAPER, tmp_path / "c.pdf")) as doc:
        assert "#set page(fill: red)" in flat(doc[1])


def test_falls_back_to_title_when_no_url(tmp_path):
    paper = PAPER.model_copy(update={"url": ""})
    with pymupdf.open(render_deck(DECK, paper, tmp_path / "c.pdf", Theme(accent="#ff0000"))) as doc:
        assert "A Test Paper" in flat(doc[0])
