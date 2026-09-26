import json

import pytest

from paper2carousel.phases import ModelSwitcher
from paper2carousel.schemas import Bullet, ClaimCard, Claims, SlideText, WrittenSlides
from paper2carousel.steps.factcheck import deterministic_problems, fact_check, numbers_in
from paper2carousel.steps.llm import LLM
from tests.conftest import agentic_chat

CARDS = [
    ClaimCard(
        id="c01",
        claim="Big model trained 3.5 days.",
        kind="result",
        section="Training",
        page=7,
        match=100,
        evidence_quote="The big models were trained for 300,000 steps (3.5 days).",
    ),
    ClaimCard(
        id="c02",
        claim="28.4 BLEU on EN-DE.",
        kind="result",
        section="Results",
        page=8,
        match=100,
        evidence_quote="achieves 28.4 BLEU on the WMT 2014 English-to-German translation task",
    ),
]
CLAIMS = Claims(cards=CARDS)


def slide(*bullets):
    return SlideText(title="Results", bullets=[Bullet(text=t, claim_ids=ids) for t, ids in bullets])


def test_numbers_in_normalises_separators():
    assert numbers_in("300,000 steps, 3.5 days, 28.4 BLEU, P100") == {
        "300000",
        "3.5",
        "28.4",
        "100",
    }


def test_deterministic_check_catches_number_drift_but_not_rewording():
    assert deterministic_problems("Trained for 3.5 days (300,000 steps).", CARDS[:1]) == []
    assert deterministic_problems("Trained for 4 days.", CARDS[:1]) == [
        "number(s) 4 not in the cited evidence"
    ]
    # evidence is the verified quote, not the (model-written) claim paraphrase
    assert deterministic_problems("28.4 BLEU", CARDS[:1]) == [
        "number(s) 28.4 not in the cited evidence"
    ]


class Recorder(ModelSwitcher):
    def __init__(self):
        super().__init__(None)
        self.order = []

    def use(self, model):
        super().use(model)
        if not self.order or self.order[-1] != model:
            self.order.append(model)


def run_loop(fake, written, max_rounds=2):
    switcher = Recorder()
    result = fact_check(
        written,
        [[b.claim_ids[0] for b in s.bullets] for s in written.slides],
        CLAIMS,
        LLM(fake.client(), "writer"),
        LLM(fake.client(), "judge"),
        switcher,
        max_rounds=max_rounds,
    )
    return result, switcher


@pytest.fixture
def model(fake):
    fake.installed.update({"writer": "w" * 64, "judge": "j" * 64})
    fake.chat_handler = agentic_chat
    return fake


def test_clean_slides_pass_in_one_round_with_one_model(model):
    written = WrittenSlides(hook="h", slides=[slide(("Trained for 3.5 days.", ["c01"]))])
    result, switcher = run_loop(model, written)
    assert len(result.report.rounds) == 1 and result.report.failed_first == 0
    assert result.slides == written and switcher.order == ["judge"]


def test_failing_bullet_is_rewritten_and_rejudged(model):
    written = WrittenSlides(
        hook="h",
        slides=[
            slide(("Trained for 3.5 days.", ["c01"])),
            slide(("WRONG: beats every model ever.", ["c02"])),
        ],
    )
    result, switcher = run_loop(model, written)
    report = result.report
    assert report.failed_first == 1 and report.total_first == 2
    assert len(report.rounds) == 2 and all(c.passed for c in report.rounds[1])
    assert result.slides.slides[0] == written.slides[0]  # untouched
    assert "WRONG" not in result.slides.slides[1].bullets[0].text
    assert switcher.order == ["judge", "writer", "judge"]  # phases, not interleaving
    judged = [b for p, b in model.requests if p == "/api/chat" and b["model"] == "judge"]
    assert len(judged) == 3  # 2 slides, then only the rewritten one


def test_bullets_still_failing_after_budget_are_dropped(model):
    def stubborn(body):
        if (
            "bullets" in body["format"]["properties"]
            and "verdicts" not in body["format"]["properties"]
        ):
            content = {
                "title": "Results",
                "bullets": [{"text": "WRONG again", "claim_ids": ["c02"]}],
            }
            return {"model": body["model"], "message": {"content": json.dumps(content)}}
        return agentic_chat(body)

    model.chat_handler = stubborn
    written = WrittenSlides(
        hook="h",
        slides=[slide(("Trained for 3.5 days.", ["c01"])), slide(("WRONG claim", ["c02"]))],
    )
    result, _ = run_loop(model, written, max_rounds=1)
    report = result.report
    assert len(report.rounds) == 2 and len(report.dropped) == 1
    assert report.dropped_slides == [2] and len(result.slides.slides) == 1


def test_number_drift_fails_even_if_the_judge_is_fooled(model):
    written = WrittenSlides(hook="h", slides=[slide(("Trained for 4 days.", ["c01"]))])
    result, _ = run_loop(model, written, max_rounds=0)
    check = result.report.rounds[0][0]
    assert check.verdict == "supported" and not check.passed
    assert result.report.dropped_slides == [1]


def test_switcher_unloads_previous_model(fake):
    client = fake.client()
    for m in ("qwen3.6:35b-mlx", "gemma4:26b-mlx"):
        fake.loaded.add(m)
    switcher = ModelSwitcher(client, unload_wait_s=0)
    switcher.use("qwen3.6:35b-mlx")
    switcher.use("qwen3.6:35b-mlx")
    switcher.use("gemma4:26b-mlx")
    assert switcher.swaps == 1 and "qwen3.6:35b-mlx" not in fake.loaded
    switcher.release()
    assert "gemma4:26b-mlx" not in fake.loaded and switcher.active is None
