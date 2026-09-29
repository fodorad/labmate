import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from labmate.config import ReplayMode
from labmate.core.callbacks import TraceHandler
from labmate.core.lc import RecordedChatModel, prompt, step, structured
from labmate.core.llm.replay import CassetteStore, ReplayClient
from labmate.core.model import LLM
from labmate.core.tracing import Tracer, read_trace


class Claim(BaseModel):
    claim: str
    page: int


@pytest.fixture
def model(fake, tmp_path):
    fake.chat_handler = lambda b: {
        "model": b["model"],
        "message": {"content": '{"claim": "x", "page": 1}'},
        "prompt_eval_count": 20,
        "eval_count": 5,
    }
    backend = ReplayClient(fake.client(), CassetteStore(tmp_path / "cassettes"), ReplayMode.AUTO)
    return RecordedChatModel(llm=LLM(backend, "qwen3.6:35b-mlx"))


def run_traced(chain, value, tmp_path, **attrs):
    tracer = Tracer(tmp_path / "trace.jsonl", trace_id="t1")
    config = {"callbacks": [TraceHandler(tracer, attrs)], "run_name": "demo"}
    return chain.invoke(value, config), read_trace(tmp_path / "trace.jsonl")


def test_a_chain_is_traced_as_run_steps_and_model_calls(model, tmp_path):
    extract = prompt("Extract from {text}") | structured(model, Claim)
    chain = step(
        "extract",
        lambda texts, config: extract.batch([{"text": t} for t in texts], config),
        summary=lambda claims: {"claims": len(claims)},
    ) | step("count", lambda claims, config: len(claims))

    result, spans = run_traced(chain, ["a", "b", "c"], tmp_path, paper="p1")

    assert result == 3
    by_name = {}
    for s in spans:
        by_name.setdefault(s["name"], []).append(s)
    assert set(by_name) == {"run", "step.extract", "step.count", "llm.chat"}  # no internals
    (root,), (extract_span,), (count_span,) = (
        by_name["run"],
        by_name["step.extract"],
        by_name["step.count"],
    )
    assert root["command"] == "demo" and root["paper"] == "p1" and root["parent_id"] is None
    assert extract_span["parent_id"] == count_span["parent_id"] == root["span_id"]
    assert extract_span["claims"] == 3  # the step's summary
    calls = by_name["llm.chat"]
    assert len(calls) == 3 and all(c["parent_id"] == extract_span["span_id"] for c in calls)
    assert all(c["model"] == "qwen3.6:35b-mlx" and c["tokens_in"] == 20 for c in calls)
    assert {c["cached"] for c in calls} == {False} and len({c["key"] for c in calls}) == 3
    assert {s["trace_id"] for s in spans} == {"t1"}


def test_a_failing_step_is_recorded_as_an_error(model, tmp_path):
    def broken(value, config):
        model.invoke([HumanMessage("hi")], config)
        raise KeyError("no such claim")

    chain = step("broken", broken) | step("never", lambda value, config: value)
    with pytest.raises(KeyError):
        run_traced(chain, "x", tmp_path)
    spans = {s["name"]: s for s in read_trace(tmp_path / "trace.jsonl")}
    assert spans["step.broken"]["status"] == spans["run"]["status"] == "error"
    assert "no such claim" in spans["step.broken"]["error"]
    assert spans["llm.chat"]["status"] == "ok" and "step.never" not in spans
