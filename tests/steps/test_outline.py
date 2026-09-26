import pytest

from paper2carousel.schemas import ClaimCard, Claims, Outline, OutlineSlide, Route
from paper2carousel.steps.llm import LLM
from paper2carousel.steps.outline import TEMPLATES, check_outline, format_claims, plan_outline
from tests.conftest import agentic_chat

CLAIMS = Claims(
    cards=[
        ClaimCard(
            id=f"c0{i}",
            claim=f"claim {i}",
            evidence_quote="q",
            kind="result",
            section="S",
            page=i,
            match=100,
        )
        for i in range(1, 5)
    ]
)


def outline(*slides):
    return Outline(
        hook="h", slides=[OutlineSlide(title="t", purpose=p, claim_ids=ids) for p, ids in slides]
    )


def test_valid_outline_has_no_problems():
    o = outline(("problem", ["c01"]), ("result", ["c02", "c03"]), ("takeaway", ["c04"]))
    assert check_outline(o, CLAIMS, "method") == []


@pytest.mark.parametrize(
    ("slides", "problem"),
    [
        (
            [("idea", ["c01"]), ("result", ["c02"]), ("takeaway", ["c03"])],
            'first slide\'s purpose must be "problem"',
        ),
        (
            [("problem", ["c01"]), ("result", ["c02"]), ("idea", ["c03"])],
            'last slide\'s purpose must be "takeaway"',
        ),
        (
            [("problem", ["c01"]), ("finding", ["c02"]), ("takeaway", ["c03"])],
            'purpose "finding" is not one of',
        ),
        (
            [("problem", []), ("result", ["c02"]), ("takeaway", ["c03"])],
            "needs 1 to 3 claim ids, has 0",
        ),
        (
            [("problem", ["c01"]), ("result", ["c09"]), ("takeaway", ["c03"])],
            "unknown claim ids ['c09']",
        ),
        (
            [("problem", ["c01"]), ("result", ["c01"]), ("takeaway", ["c01"])],
            "more than 2 slides: ['c01']",
        ),
    ],
)
def test_each_rule_is_enforced(slides, problem):
    problems = check_outline(outline(*slides), CLAIMS, "method")
    assert any(problem in p for p in problems), problems


def test_every_template_starts_somewhere_and_ends_with_takeaway():
    assert all(t[-1] == "takeaway" for t in TEMPLATES.values())


def test_format_claims_lists_ids_kinds_and_pages():
    assert format_claims(CLAIMS).splitlines()[0] == "c01 [result] claim 1 (S, p. 1)"


def test_planner_gets_rule_violations_fed_back(fake):
    replies = iter(
        [
            '{"hook": "h", "slides": [{"title": "t", "purpose": "problem", "claim_ids": ["c01"]},'
            ' {"title": "t", "purpose": "result", "claim_ids": ["c02"]},'
            ' {"title": "t", "purpose": "result", "claim_ids": ["c03"]}]}',
        ]
    )

    def model(body):
        try:
            return {"model": body["model"], "message": {"content": next(replies)}}
        except StopIteration:
            return agentic_chat(body)

    fake.chat_handler = model
    route = Route(paper_type="method", confidence=0.9, reason="r")
    result = plan_outline("T", route, CLAIMS, LLM(fake.client(), "qwen3.6:35b-mlx"))
    assert result.slides[-1].purpose == "takeaway"
    retry = fake.requests[1][1]["messages"][-1]["content"]
    assert 'last slide\'s purpose must be "takeaway"' in retry


def test_planning_without_claims_is_an_error(fake):
    route = Route(paper_type="method", confidence=0.9, reason="r")
    with pytest.raises(ValueError, match="no verified claim cards"):
        plan_outline("T", route, Claims(cards=[]), LLM(fake.client(), "m"))
