from paper2carousel.schemas import Bullet, Paper, SlideText, WrittenSlides
from paper2carousel.steps.summary import summary_markdown
from tests.steps.test_factcheck import CLAIMS


def test_every_bullet_carries_its_page_and_quote():
    slides = WrittenSlides(
        hook="Hook",
        slides=[
            SlideText(
                title="Results", bullets=[Bullet(text="28.4 BLEU.", claim_ids=["c02", "c77"])]
            )
        ],
    )
    text = summary_markdown(slides, CLAIMS, Paper(paper_id="x", title="T", url="https://u"))
    assert text.splitlines() == [
        "# Hook",
        "",
        "Summary of *T* (https://u)",
        "",
        "## 1. Results",
        "",
        "- 28.4 BLEU.",
        '  - p. 8 (Results): "achieves 28.4 BLEU on the WMT 2014 English-to-German '
        'translation task"',
    ]
