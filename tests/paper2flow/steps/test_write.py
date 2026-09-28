from labmate.core.factcheck import check_slide
from labmate.paper2flow.schemas import Bullet, SlideText


def slide(title="Short title", *bullets):
    return SlideText(title=title, bullets=list(bullets) or [Bullet(text="ok", claim_ids=["c01"])])


def test_valid_slide():
    assert check_slide(slide(), {"c01"}) == []


def test_long_title_long_bullet_and_foreign_claims_are_flagged():
    problems = check_slide(
        slide(
            "one two three four five six seven eight nine ten eleven",
            Bullet(text="word " * 31, claim_ids=["c01"]),
            Bullet(text="fine", claim_ids=["c01", "c07"]),
        ),
        {"c01"},
    )
    assert problems == [
        "the title has more than 10 words",
        "bullet 1 has more than 30 words",
        "bullet 2 cites claims not on this slide: ['c07']",
    ]


def test_claim_ids_in_the_text_are_flagged():
    problems = check_slide(
        slide("t", Bullet(text="Faster training [c03].", claim_ids=["c01"])), {"c01"}
    )
    assert problems == ["bullet 1 has claim ids in its text; list them in claim_ids only"]
