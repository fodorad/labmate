from paper2carousel.schemas import Bullet, SlideText
from paper2carousel.steps.write import check_slide


def slide(title="Short title", *bullets):
    return SlideText(title=title, bullets=list(bullets) or [Bullet(text="ok", claim_ids=["c01"])])


def test_valid_slide():
    assert check_slide(slide(), {"c01"}) == []


def test_long_title_long_bullet_and_foreign_claims_are_flagged():
    problems = check_slide(
        slide(
            "one two three four five six seven eight nine ten eleven",
            Bullet(text="word " * 30, claim_ids=["c01"]),
            Bullet(text="fine", claim_ids=["c01", "c07"]),
        ),
        {"c01"},
    )
    assert problems == [
        "the title has more than 10 words",
        "bullet 1 has more than 25 words",
        "bullet 2 cites claims not on this slide: ['c07']",
    ]
