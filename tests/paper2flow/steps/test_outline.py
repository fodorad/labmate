import pytest

from labmate.paper2flow.schemas import ClaimCard, Claims, Outline, PlannedCard, Route
from labmate.paper2flow.steps.outline import check_outline, plan_outline
from tests.conftest import agentic_chat, chat_model

CLAIMS = Claims(
    cards=[
        ClaimCard(
            id=f"c0{i}",
            claim=f"claim {i}",
            evidence_quote="q",
            kind="result",
            section="S",
            page=i,
        )
        for i in range(1, 5)
    ]
)


def outline(*cards):
    return Outline(cards=[PlannedCard(title="t", purpose=p, claim_ids=ids) for p, ids in cards])


FOUR = ["task", "challenges", "method", "results"]


def blocks(*ids):
    return outline(*zip(FOUR, ids, strict=True))


def test_valid_outline_has_no_problems():
    o = blocks(["c01"], ["c02", "c03"], ["c03", "c04"], ["c04"])
    assert check_outline(o, CLAIMS) == []


@pytest.mark.parametrize(
    ("cards", "problem"),
    [
        (
            [("task", ["c01"]), ("method", ["c02"]), ("results", ["c03"]), ("method", ["c04"])],
            "the cards must be exactly ['task', 'challenges', 'method', 'results']",
        ),
        (
            [("challenges", ["c01"]), ("task", ["c02"]), ("method", ["c03"]), ("results", ["c04"])],
            "in this order",
        ),
        (
            list(zip(FOUR, [[], ["c02"], ["c03"], ["c04"]], strict=True)),
            "needs 1 to 5 claim ids, has 0",
        ),
        (
            list(zip(FOUR, [["c01"], ["c09"], ["c03"], ["c04"]], strict=True)),
            "unknown claim ids ['c09']",
        ),
        (
            list(zip(FOUR, [["c01"], ["c01"], ["c01"], ["c04"]], strict=True)),
            "more than 2 cards: ['c01']",
        ),
    ],
)
def test_each_rule_is_enforced(cards, problem):
    problems = check_outline(outline(*cards), CLAIMS)
    assert any(problem in p for p in problems), problems


def test_planner_gets_rule_violations_fed_back(fake):
    replies = iter(
        [
            '{"cards": [{"title": "t", "purpose": "task", "claim_ids": ["c01"]},'
            ' {"title": "t", "purpose": "results", "claim_ids": ["c02"]},'
            ' {"title": "t", "purpose": "method", "claim_ids": ["c03"]},'
            ' {"title": "t", "purpose": "challenges", "claim_ids": ["c04"]}]}',
        ]
    )

    def model(body):
        try:
            return {"model": body["model"], "message": {"content": next(replies)}}
        except StopIteration:
            return agentic_chat(body)

    fake.chat_handler = model
    route = Route(paper_type="method", confidence=0.9, reason="r")
    result = plan_outline("T", route, CLAIMS, chat_model(fake))
    assert [c.purpose for c in result.cards] == FOUR
    retry = fake.requests[1][1]["messages"][-1]["content"]
    assert "the cards must be exactly" in retry


def test_planning_without_claims_is_an_error(fake):
    route = Route(paper_type="method", confidence=0.9, reason="r")
    with pytest.raises(ValueError, match="no verified claim cards"):
        plan_outline("T", route, Claims(cards=[]), chat_model(fake))
