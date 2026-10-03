import json

from labmate.core.schemas import Bullet, Card, Cards, Paper, Section
from labmate.paper2post.post import STORY_ICONS, post_links, write_post
from tests.conftest import agentic_chat, chat, reply
from tests.core.test_factcheck import CLAIMS

CARDS = Cards(
    cards=[Card(title="R", bullets=[Bullet(text="28.4 BLEU on EN-DE.", claim_ids=["c02"])])]
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
            content = json.dumps(next(replies))
            return reply(body, content)
        return agentic_chat(body)

    fake.chat_handler = model
    post = write_post(
        CARDS,
        CLAIMS,
        PAPER,
        chat(fake, "qwen3.6:35b-mlx"),
        chat(fake, "gemma4:26b-mlx"),
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

    assert post.icons == STORY_ICONS[:3]  # one per surviving sentence, by place in the story


def test_links_are_the_paper_its_code_and_yours():
    paper = PAPER.model_copy(
        update={"sections": [Section(title="Code", page=9, text="See https://github.com/a/b.")]}
    )
    links = post_links(paper, {"Website": "https://adamfodor.com"})
    assert [(x.label, x.url) for x in links] == [
        ("Paper", "https://arxiv.org/abs/1706.03762"),
        ("Code", "https://github.com/a/b"),
        ("Website", "https://adamfodor.com"),
    ]
    assert post_links(PAPER.model_copy(update={"url": ""}), {}) == []
