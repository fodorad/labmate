import pytest

from paper2flow.schemas import ClaimCard, Claims, Outline, OutlineSlide, Route
from paper2flow.steps.llm import LLM
from paper2flow.steps.outline import TEMPLATES, check_outline, format_claims, plan_outline
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


FOUR = ["task", "challenges", "method", "results"]


def blocks(*ids):
    return outline(*zip(FOUR, ids, strict=True))


def test_valid_outline_has_no_problems():
    o = blocks(["c01"], ["c02", "c03"], ["c03", "c04"], ["c04"])
    assert check_outline(o, CLAIMS, "method") == []
    assert check_outline(o, CLAIMS, "survey") == []  # every paper type uses the four blocks


@pytest.mark.parametrize(
    ("slides", "problem"),
    [
        (
            [("task", ["c01"]), ("method", ["c02"]), ("results", ["c03"])],
            "the slides must be exactly ['task', 'challenges', 'method', 'results']",
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
            "more than 2 slides: ['c01']",
        ),
    ],
)
def test_each_rule_is_enforced(slides, problem):
    problems = check_outline(outline(*slides), CLAIMS, "method")
    assert any(problem in p for p in problems), problems


def test_every_paper_type_gets_the_four_blocks():
    assert all(t == FOUR for t in TEMPLATES.values())


def test_format_claims_lists_ids_kinds_and_pages():
    assert format_claims(CLAIMS).splitlines()[0] == "c01 [result] claim 1 (S, p. 1)"


def test_planner_gets_rule_violations_fed_back(fake):
    replies = iter(
        [
            '{"hook": "h", "slides": [{"title": "t", "purpose": "task", "claim_ids": ["c01"]},'
            ' {"title": "t", "purpose": "method", "claim_ids": ["c02"]},'
            ' {"title": "t", "purpose": "results", "claim_ids": ["c03"]}]}',
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
    assert [s.purpose for s in result.slides] == FOUR
    retry = fake.requests[1][1]["messages"][-1]["content"]
    assert "the slides must be exactly" in retry


def test_planning_without_claims_is_an_error(fake):
    route = Route(paper_type="method", confidence=0.9, reason="r")
    with pytest.raises(ValueError, match="no verified claim cards"):
        plan_outline("T", route, Claims(cards=[]), LLM(fake.client(), "m"))
