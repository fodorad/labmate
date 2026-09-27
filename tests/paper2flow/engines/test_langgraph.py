"""The LangGraph engine: same artifacts as the plain engine, plus its own pause/resume."""

import json
import shutil

import pytest
import yaml

from labmate.config import Config, ReplayMode
from labmate.core.tracing import read_trace
from labmate.paper2flow.engines import langgraph_engine, plain
from labmate.paper2flow.engines.common import NothingSupportedError
from tests.conftest import FakeArxiv, agentic_chat, make_pdf

REF = "2401.00001"
COMPARED = [
    "00_publication.json",
    "01_route.json",
    "02_claims.json",
    "03_outline.draft.json",
    "03_outline.json",
    "04_slides.json",
    "05_factcheck.json",
    "08_post.json",
    "09_flows.json",
    "overview.pdf",
    "post.pdf",
]


def flaky_writer(body):
    """Block 2's first draft is unsupported; the rewrite fixes it (exercises the cycle)."""
    reply = agentic_chat(body)
    prompt = body["messages"][-1]["content"]
    writing = "bullets" in body.get("format", {}).get("properties", {})
    if writing and "Block 2 of" in prompt and "flagged problems" not in prompt:
        content = json.loads(reply["message"]["content"])
        content["bullets"][0]["text"] = "WRONG: better than everything."
        reply["message"]["content"] = json.dumps(content)
    return reply


def config_for(tmp_path, name):
    cfg = Config()
    cfg.replay.dir = tmp_path / "cassettes"
    cfg.replay.lock_file = tmp_path / "models.lock"
    cfg.tracing.runs_dir = tmp_path / name
    return cfg


@pytest.fixture
def paper(tmp_path):
    return FakeArxiv(make_pdf(tmp_path / "src.pdf", figure=True).read_bytes())


@pytest.fixture
def model(fake):
    fake.chat_handler = flaky_writer
    return fake


def test_both_engines_write_identical_artifacts_in_replay(model, paper, tmp_path):
    cfg = config_for(tmp_path, "plain")
    plain.run(cfg, ref=REF, client=model.client(), http=paper.client())
    done = plain.run(cfg, ref=REF, client=model.client(), http=paper.client(), approve=True)
    rounds = json.loads(done.artifact("05_factcheck.json").read_text())["report"]["rounds"]
    assert len(rounds) == 2  # the rewrite cycle ran

    graph_cfg = config_for(tmp_path, "graph")
    graph_dir = graph_cfg.tracing.runs_dir / REF
    graph_dir.mkdir(parents=True)
    shutil.copy2(done.artifact("03_outline.json"), graph_dir / "03_outline.json")
    result = langgraph_engine.run(
        graph_cfg, ref=REF, mode=ReplayMode.REPLAY, http=paper.client(), approve=True
    )
    assert result.status == "done" and result.overview.exists()
    for name in COMPARED:
        assert (graph_dir / name).read_bytes() == done.artifact(name).read_bytes(), name

    spans = read_trace(result.trace)
    names = {s["name"] for s in spans}
    assert {"step.extract.unit", "step.write.slide", "step.factcheck.judge"} <= names
    assert {"step.flow.overview", "step.flow.detail", "step.flows", "step.render"} <= names
    assert "step.factcheck.rewrite" in names
    assert all(s.get("cached") for s in spans if s["name"] == "llm.chat")
    root = next(s for s in spans if s["name"] == "run")
    assert root["engine"] == "langgraph" and root["status"] == "ok"
    # extract/write fan-out spans hang under the run span, like the plain engine's steps
    units = [s for s in spans if s["name"] == "step.extract.unit"]
    assert units and {s["parent_id"] for s in units} == {root["span_id"]}


def test_interrupt_pauses_at_the_gate_and_resumes_from_the_checkpoint(model, paper, tmp_path):
    cfg = config_for(tmp_path, "graph")
    first = langgraph_engine.run(cfg, ref=REF, client=model.client(), http=paper.client())
    assert first.status == "awaiting_approval" and first.gate.exists()
    assert (first.run_dir / "langgraph.sqlite").exists()
    assert not first.artifact("04_slides.json").exists()
    root = next(s for s in read_trace(first.trace) if s["name"] == "run")
    assert root["status"] == "awaiting_approval"

    edited = yaml.safe_load(first.gate.read_text())
    edited["hook"] = "Edited in the gate"
    first.gate.write_text(yaml.safe_dump(edited))
    calls = model.paths().count("/api/chat")

    done = langgraph_engine.run(
        cfg, ref=REF, client=model.client(), http=paper.client(), approve=True
    )
    assert done.status == "done" and done.overview.exists() and done.post.exists()
    written = json.loads(done.artifact("04_slides.json").read_text())
    assert written["hook"] == "Edited in the gate"
    resumed = [s for s in read_trace(done.trace) if s["trace_id"] == done.trace_id]
    names = [s["name"] for s in resumed]
    assert "step.route" not in names and "step.outline" not in names  # resumed, not rerun
    assert model.paths().count("/api/chat") > calls


def test_auto_approve_and_fresh(model, paper, tmp_path):
    cfg = config_for(tmp_path, "graph")
    first = langgraph_engine.run(
        cfg, ref=REF, client=model.client(), http=paper.client(), auto_approve=True
    )
    assert first.status == "done" and not first.gate.exists()
    again = langgraph_engine.run(
        cfg, ref=REF, mode=ReplayMode.REPLAY, http=paper.client(), fresh=True
    )
    assert again.status == "done"  # approved outline reused, everything else replayed


def test_nothing_supported_propagates(fake, paper, tmp_path):
    def all_wrong(body):
        reply = agentic_chat(body)
        if "bullets" in body.get("format", {}).get("properties", {}):
            content = json.loads(reply["message"]["content"])
            for b in content["bullets"]:
                b["text"] = "WRONG"
            reply["message"]["content"] = json.dumps(content)
        return reply

    fake.chat_handler = all_wrong
    cfg = config_for(tmp_path, "graph")
    with pytest.raises(NothingSupportedError):
        langgraph_engine.run(
            cfg, ref=REF, client=fake.client(), http=paper.client(), auto_approve=True
        )


def test_an_overview_without_steps_to_expand_skips_the_details(fake, paper, tmp_path):
    def no_details(body):
        reply = agentic_chat(body)
        if "expand" in body.get("format", {}).get("properties", {}):
            content = json.loads(reply["message"]["content"])
            content["nodes"], content["edges"] = content["nodes"][:3], content["edges"][:2]
            content["nodes"][2]["kind"] = "output"
            content["expand"] = []
            reply["message"]["content"] = json.dumps(content)
        return reply

    fake.chat_handler = no_details
    cfg = config_for(tmp_path, "graph")
    result = langgraph_engine.run(
        cfg, ref=REF, client=fake.client(), http=paper.client(), auto_approve=True
    )
    flows = json.loads(result.artifact("09_flows.json").read_text())
    assert result.status == "done" and flows["details"] == []
