import pytest

from labmate.paper2flow.schemas import ClaimCard, Claims
from labmate.paper2flow.steps.write import MAX_CARD_CLAIMS, plan_cards, write_cards
from tests.conftest import agentic_chat, chat


def claim(i: int, kind: str, quote: str = "") -> ClaimCard:
    return ClaimCard(
        id=f"c{i:02d}", claim=f"Claim {i}.", evidence_quote=quote or f"Quote {i}.", kind=kind,
        section="S", page=1,
    )  # fmt: skip


def test_claims_are_grouped_into_the_four_cards_by_kind():
    kinds = ["task", "result", "limitation", "contribution", "challenge", "method"]
    claims = Claims(cards=[claim(i, kind) for i, kind in enumerate(kinds, start=1)])

    outline = plan_cards(claims)

    assert [(c.purpose, c.claim_ids) for c in outline.cards] == [
        ("task", ["c01"]),
        ("challenges", ["c03", "c05"]),
        ("method", ["c04", "c06"]),
        ("results", ["c02"]),
    ]


def test_a_card_without_claims_is_left_out_and_each_card_is_capped():
    many = [claim(i, "result") for i in range(1, 12)]

    outline = plan_cards(Claims(cards=[claim(0, "method"), *many]))

    assert [c.purpose for c in outline.cards] == ["method", "results"]
    assert len(outline.cards[1].claim_ids) == MAX_CARD_CLAIMS


def test_results_with_numbers_come_before_the_rest():
    claims = Claims(cards=[
        claim(1, "result", "The model is better."),
        claim(2, "result", "It reaches 28.4 BLEU on WMT."),
        claim(3, "result", "It is simple."),
        claim(4, "result", "It trains in 3.5 days."),
    ])  # fmt: skip

    (card,) = plan_cards(claims).cards

    assert card.claim_ids == ["c02", "c04", "c01", "c03"]


def test_no_claims_means_nothing_to_write():
    with pytest.raises(ValueError, match="nothing to build the overview from"):
        plan_cards(Claims(cards=[]))


def test_each_card_is_written_from_its_own_claims_and_titled_by_the_writer(fake):
    fake.chat_handler = agentic_chat
    claims = Claims(cards=[claim(1, "task"), claim(2, "result")])

    cards = write_cards(plan_cards(claims), claims, chat(fake), workers=1)

    assert [c.title for c in cards.cards] == ["A written card", "A written card"]
    prompts = [b["messages"][-1]["content"] for p, b in fake.requests if p == "/api/chat"]
    assert 'This is the "Task" card' in prompts[0] and "c02" not in prompts[0]
    assert (
        'This is the "Main results" card' in prompts[1]
        and "names the task or dataset" in prompts[1]
    )
