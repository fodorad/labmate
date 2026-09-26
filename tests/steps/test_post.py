import json

from paper2carousel.phases import ModelSwitcher
from paper2carousel.schemas import Bullet, Paper, SlideText, WrittenSlides
from paper2carousel.steps.llm import LLM
from paper2carousel.steps.post import post_markdown, write_post
from tests.conftest import agentic_chat
from tests.steps.test_factcheck import CLAIMS

SLIDES = WrittenSlides(
    hook="h",
    slides=[SlideText(title="R", bullets=[Bullet(text="28.4 BLEU on EN-DE.", claim_ids=["c02"])])],
)
PAPER = Paper(
    paper_id="x", title="Attention Is All You Need", url="https://arxiv.org/abs/1706.03762"
)


def test_post_takeaways_are_fact_checked_and_unknown_claims_rejected(fake):
    replies = iter(
        [
            {
                "hook": "One idea",
                "takeaways": [
                    {"text": "x", "claim_ids": ["c99"]},
                    {"text": "y", "claim_ids": ["c02"]},
                    {"text": "z", "claim_ids": ["c02"]},
                ],
                "question": "q?",
            },
            {
                "hook": "One idea",
                "takeaways": [
                    {"text": "28.4 BLEU.", "claim_ids": ["c02"]},
                    {"text": "WRONG: 99 BLEU.", "claim_ids": ["c02"]},
                    {"text": "It uses attention.", "claim_ids": ["c02"]},
                    {"text": "It trains fast.", "claim_ids": ["c02"]},
                ],
                "question": "What do you think?",
            },
        ]
    )

    def model(body):
        if "takeaways" in body["format"]["properties"]:
            return {"model": body["model"], "message": {"content": json.dumps(next(replies))}}
        return agentic_chat(body)

    fake.chat_handler = model
    fake.installed.update({"w": "w" * 64, "j": "j" * 64})
    post = write_post(
        SLIDES,
        CLAIMS,
        PAPER,
        LLM(fake.client(), "w"),
        LLM(fake.client(), "j"),
        ModelSwitcher(None),
        max_rounds=0,
    )
    retry = fake.requests[1][1]["messages"][-1]["content"]
    assert "takeaway 1 cites unknown claims ['c99']" in retry
    # checked in groups of three, order kept, the made-up one dropped
    assert [t.text for t in post.takeaways] == [
        "28.4 BLEU.",
        "It uses attention.",
        "It trains fast.",
    ]
    assert post.report.failed_first == 1


def test_post_markdown():
    from paper2carousel.schemas import FactCheckReport, Post

    post = Post(
        hook="H",
        takeaways=[Bullet(text="T1", claim_ids=["c01"])],
        question="Q?",
        report=FactCheckReport(rounds=[]),
    )
    text = post_markdown(post, PAPER)
    assert text.splitlines()[:4] == ["H", "", "T1", ""]
    assert "https://arxiv.org/abs/1706.03762" in text
    assert (
        "https://"
        not in post_markdown(post, PAPER.model_copy(update={"url": ""}))
        .split("Paper:")[1]
        .split("\n")[0]
    )
