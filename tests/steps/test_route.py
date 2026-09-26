import json

from paper2carousel.schemas import Paper
from paper2carousel.steps.llm import LLM
from paper2carousel.steps.route import route_paper

PAPER = Paper(paper_id="x", title="A Survey of Things", abstract="We review things.")


def routed(fake, paper_type, confidence):
    reply = {"paper_type": paper_type, "confidence": confidence, "reason": "because"}
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": json.dumps(reply)}}
    return route_paper(PAPER, LLM(fake.client(), "qwen3.6:35b-mlx"))


def test_confident_route_is_kept(fake):
    assert routed(fake, "survey", 0.9).paper_type == "survey"


def test_unsure_route_falls_back_to_method_and_says_why(fake):
    route = routed(fake, "survey", 0.4)
    assert route.paper_type == "method"
    assert route.reason.startswith("fallback from survey (confidence 0.40)")


def test_only_title_and_abstract_are_sent(fake):
    routed(fake, "method", 0.9)
    prompt = fake.requests[0][1]["messages"][-1]["content"]
    assert "A Survey of Things" in prompt and "We review things." in prompt
