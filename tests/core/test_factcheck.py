import json

import pytest

from labmate.core.factcheck import (
    check_card,
    deterministic_problems,
    fact_check,
    names_in,
    numbers_in,
    title_names,
)
from labmate.core.schemas import Bullet, Card, Cards, ClaimCard, Claims
from tests.conftest import agentic_chat, chat, reply

CARDS = [
    ClaimCard(
        id="c01",
        claim="Big model trained 3.5 days.",
        kind="result",
        section="Training",
        page=7,
        evidence_quote="The big models were trained for 300,000 steps (3.5 days).",
    ),
    ClaimCard(
        id="c02",
        claim="28.4 BLEU on EN-DE.",
        kind="result",
        section="Results",
        page=8,
        evidence_quote="achieves 28.4 BLEU on the WMT 2014 English-to-German translation task",
    ),
]
CLAIMS = Claims(cards=CARDS)


def card(*bullets):
    return Card(title="Results", bullets=[Bullet(text=t, claim_ids=ids) for t, ids in bullets])


def test_numbers_in_normalises_separators():
    assert numbers_in("300,000 steps, 3.5 days, 28.4 BLEU, P100") == {"300000", "3.5", "28.4"}


def test_digits_inside_names_are_not_numbers():
    # a real run dropped "F1 score of 0.744" because "F1" counted as the number 1
    assert numbers_in("F1 score of 0.744 for ResNet50 on 64x64 patches") == {"0.744", "64"}


def test_deterministic_check_catches_number_drift_but_not_rewording():
    assert deterministic_problems("Trained for 3.5 days (300,000 steps).", CARDS[:1]) == []
    assert deterministic_problems("Trained for 4 days.", CARDS[:1]) == [
        "number(s) 4 not in the cited evidence"
    ]
    # evidence is the verified quote, not the (model-written) claim paraphrase
    assert deterministic_problems("28.4 BLEU", CARDS[:1]) == [
        "number(s) 28.4 not in the cited evidence"
    ]


WRITER, JUDGE = "qwen3.8:27b-mlx", "gemma4:26b-mlx"


def run_loop(fake, written, max_rounds=2):
    scopes = [[b.claim_ids[0] for b in c.bullets] for c in written.cards]
    return fact_check(
        written, scopes, CLAIMS, chat(fake, WRITER), chat(fake, JUDGE), max_rounds=max_rounds
    )


def models_called(fake):
    """The model of every chat request, consecutive repeats collapsed."""
    called = [b["model"] for p, b in fake.requests if p == "/api/chat"]
    return [m for i, m in enumerate(called) if i == 0 or m != called[i - 1]]


@pytest.fixture
def model(fake):
    fake.chat_handler = agentic_chat
    return fake


def test_clean_cards_pass_in_one_round_with_one_model(model):
    written = Cards(cards=[card(("Trained for 3.5 days.", ["c01"]))])
    result = run_loop(model, written)
    assert len(result.report.rounds) == 1 and result.report.failed_first == 0
    assert result.cards == written and models_called(model) == [JUDGE]


def test_failing_bullet_is_rewritten_and_rejudged(model):
    written = Cards(
        cards=[
            card(("Trained for 3.5 days.", ["c01"])),
            card(("WRONG: beats every model ever.", ["c02"])),
        ]
    )
    result = run_loop(model, written)
    report = result.report
    assert report.failed_first == 1 and report.total_first == 2
    assert len(report.rounds) == 2 and all(c.passed for c in report.rounds[1])
    assert result.cards.cards[0] == written.cards[0]  # untouched
    assert "WRONG" not in result.cards.cards[1].bullets[0].text
    assert models_called(model) == [JUDGE, WRITER, JUDGE]  # phases, not interleaving
    judged = [b for p, b in model.requests if p == "/api/chat" and b["model"] == JUDGE]
    assert len(judged) == 3  # 2 cards, then only the rewritten one


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
            return reply(body, json.dumps(content))  # fmt: skip
        return agentic_chat(body)

    model.chat_handler = stubborn
    written = Cards(
        cards=[card(("Trained for 3.5 days.", ["c01"])), card(("WRONG claim", ["c02"]))]
    )
    result = run_loop(model, written, max_rounds=1)
    report = result.report
    assert len(report.rounds) == 2 and len(report.dropped) == 1
    assert report.dropped_cards == [2] and len(result.cards.cards) == 1


def test_number_drift_fails_even_if_the_judge_is_fooled(model):
    written = Cards(cards=[card(("Trained for 4 days.", ["c01"]))])
    result = run_loop(model, written, max_rounds=0)
    check = result.report.rounds[0][0]
    assert check.verdict == "supported" and not check.passed
    assert result.report.dropped_cards == [1]


def test_names_must_be_in_the_evidence_unless_generic_or_in_the_title():
    assert names_in("DenseNet121 gets 0.99 F1 on MRL, EN-DE and TalkingFace data") == {
        "densenet121",
        "f1",
        "mrl",
        "en-de",
        "talkingface",
    }
    quote = [
        CARDS[0].model_copy(
            update={"evidence_quote": "a 0.9953 average F1 score is successfully reproduced"}
        )
    ]
    bullet = "DenseNet121 achieves a 0.9953 average F1 score on the MRL database."
    assert deterministic_problems(bullet, quote) == [
        "name(s) densenet121, mrl not in the cited evidence"
    ]
    exempt = title_names("BlinkLinMulT: Transformer-Based Eye Blink Detection")
    assert "blinklinmult" in exempt
    assert deterministic_problems("BlinkLinMulT reaches 0.9953 F1.", quote, exempt) == []


def test_card_rules_flag_long_text_foreign_claims_and_inline_ids():
    assert check_card(card(("ok", ["c01"])), {"c01"}) == []
    long_title = Card(
        title="one two three four five six seven eight nine ten eleven",
        bullets=[
            Bullet(text="word " * 31, claim_ids=["c01"]),
            Bullet(text="fine", claim_ids=["c01", "c07"]),
            Bullet(text="Faster training [c03].", claim_ids=["c01"]),
        ],
    )
    assert check_card(long_title, {"c01"}) == [
        "the title has more than 10 words",
        "bullet 1 has more than 30 words",
        "bullet 2 cites claims not on this card: ['c07']",
        "bullet 3 has claim ids in its text; list them in claim_ids only",
    ]
