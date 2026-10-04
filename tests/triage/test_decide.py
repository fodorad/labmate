import json

from labmate.core.ingest import ArxivMetadata
from labmate.triage.decide import decide
from tests.conftest import reply

META = ArxivMetadata(title="Alpha", authors=["Ada"], abstract="We study alpha.")


def answers(*contents: str):
    queue = list(contents)
    return lambda body: reply(body, queue.pop(0))


def test_a_reply_that_is_too_wordy_is_sent_back_for_a_shorter_reason(fake, config):
    wordy = json.dumps({"action": "deep", "relevance": 4, "reason": "word " * 40})
    short = json.dumps({"action": "deep", "relevance": 4, "reason": "Short."})
    fake.chat_handler = answers(wordy, short)

    decision = decide(config, META, "video", transport=fake.transport())

    assert decision.reason == "Short." and len(fake.requests) == 2
