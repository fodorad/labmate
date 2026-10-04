import json

from labmate.core.ingest import ArxivMetadata
from labmate.triage.decide import decide, decision_from_score
from tests.conftest import reply

META = ArxivMetadata(title="Alpha", authors=["Ada"], abstract="We study alpha.")


def answers(*contents: str):
    queue = list(contents)
    return lambda body: reply(body, queue.pop(0))


def test_a_reply_that_is_too_wordy_is_sent_back_for_a_shorter_reason(fake, config):
    wordy = json.dumps({"action": "deep", "relevance": 4, "reason": "word " * 40})
    short = json.dumps({"action": "deep", "relevance": 4, "reason": "Short."})
    fake.chat_handler = answers(wordy, short)

    decision = decide(config, META, "video", decision_model=False, transport=fake.transport())

    assert decision.reason == "Short." and len(fake.requests) == 2


def test_a_score_becomes_an_action_at_the_documented_levels():
    assert decision_from_score(3.0).action == "deep"
    assert decision_from_score(2.9).action == "post" and decision_from_score(1.75).action == "post"
    skipped = decision_from_score(1.7)
    assert skipped.action == "skip" and skipped.relevance == 3
    assert decision_from_score(0.0).relevance == 1 and decision_from_score(4.0).relevance == 5
    assert "closely related" in decision_from_score(3.2).reason
