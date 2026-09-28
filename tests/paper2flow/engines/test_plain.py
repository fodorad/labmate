import json
from pathlib import Path

import pymupdf
import pytest
import yaml

from labmate.config import Config, ReplayMode
from labmate.core.llm.replay import CassetteMissError
from labmate.core.tracing import read_trace
from labmate.paper2flow.engines.plain import NothingSupportedError, run, run_id_for
from labmate.paper2flow.schemas import Claims, Flows, Outline, WrittenSlides
from labmate.paper2flow.steps.gate import GateError
from tests.conftest import agentic_chat, make_pdf

REF = "2401.00001"


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.replay.dir = tmp_path / "cassettes"
    cfg.replay.lock_file = tmp_path / "models.lock"
    cfg.tracing.runs_dir = tmp_path / "runs"
    return cfg


@pytest.fixture
def model(fake):
    fake.chat_handler = agentic_chat
    return fake


def n_chats(fake):
    return fake.paths().count("/api/chat")


def go(config, model, arxiv, **kw):
    return run(config, ref=REF, client=model.client(), http=arxiv.client(), **kw)


def test_run_id():
    assert run_id_for("arXiv:1706.03762v7", None) == "1706.03762"
    assert run_id_for(None, Path("My Paper.pdf")) == "my-paper"
    with pytest.raises(ValueError):
        run_id_for(None, None)


# --- agentic pipeline ----------------------------------------------------------------------


def test_first_run_pauses_at_the_gate_with_an_editable_outline(model, arxiv, config):
    result = go(config, model, arxiv)
    assert result.status == "awaiting_approval"
    assert not result.overview.exists() and not result.post.exists()
    text = result.gate.read_text()
    assert "make approve ARXIV=2401.00001" in text and "# Available claim cards:" in text
    outline = Outline.model_validate(yaml.safe_load(text))
    assert [s.purpose for s in outline.slides] == ["task", "challenges", "method", "results"]
    claims = Claims.model_validate_json(result.artifact("02_claims.json").read_text())
    assert len(claims.cards) == 6 and all(c.match >= 90 for c in claims.cards)
    assert "qwen3.6:35b-mlx" in model.loaded  # not unloaded: the run is paused


def test_approve_finishes_and_respects_human_edits(model, arxiv, config):
    first = go(config, model, arxiv)
    edited = yaml.safe_load(first.gate.read_text())
    edited["hook"] = "Edited by a human"
    edited["slides"][1]["title"] = "Why it is hard"
    first.gate.write_text(yaml.safe_dump(edited))

    result = go(config, model, arxiv, approve=True)
    assert result.status == "done"
    written = WrittenSlides.model_validate_json(result.artifact("04_slides.json").read_text())
    assert written.hook == "Edited by a human" and len(written.slides) == 4
    with pymupdf.open(result.overview) as doc:
        # paper, four blocks, data flow, two detail flows
        assert doc.page_count == 5
        assert "A Test Paper" in doc[0].get_text()
        blocks = " ".join(doc[1].get_text().split())
        assert all(
            k in blocks for k in ["Task", "Why it is hard", "Proposed method", "Main results"]
        )
        assert "END-TO-END DATA FLOW" in doc[2].get_text()
        assert "DETAIL A" in doc[3].get_text() and "DETAIL B" in doc[4].get_text()
        assert not doc.metadata["creationDate"]  # no timestamp: same run, same bytes
    with pymupdf.open(result.post) as doc:
        assert doc.page_count == 4  # the text, then one image per diagram
        assert "Where would linear attention help your models?" in doc[0].get_text()
    gate_span = next(
        s
        for s in read_trace(result.trace)
        if s["name"] == "step.gate" and s.get("status") == "approved"
    )
    assert gate_span["edited"] is True
    assert not model.loaded


def test_invalid_edit_is_rejected_with_the_rule_it_breaks(model, arxiv, config):
    first = go(config, model, arxiv)
    edited = yaml.safe_load(first.gate.read_text())
    edited["slides"][0]["claim_ids"] = ["c99"]
    first.gate.write_text(yaml.safe_dump(edited))
    with pytest.raises(GateError, match=r"unknown claim ids \['c99'\]"):
        go(config, model, arxiv, approve=True)


def test_auto_approve_runs_straight_through_and_traces_every_step(model, arxiv, config):
    result = go(config, model, arxiv, auto_approve=True)
    assert result.status == "done" and result.overview.exists() and result.post.exists()
    spans = read_trace(result.trace)
    names = {s["name"] for s in spans}
    assert {
        "step.route",
        "step.extract",
        "step.outline",
        "step.gate",
        "step.write",
        "step.factcheck",
        "step.publication",
        "step.post",
        "step.flows",
        "step.flow.overview",
        "step.flow.detail",
        "step.render",
    } <= names
    extract = next(s for s in spans if s["name"] == "step.extract")
    llm_in_extract = [
        s for s in spans if s["name"] == "llm.chat" and s["parent_id"] == extract["span_id"]
    ]
    assert len(llm_in_extract) == 3  # one per section, parent kept across worker threads
    factcheck = next(s for s in spans if s["name"] == "step.factcheck")
    assert factcheck["failed_first"] == 0 and factcheck["rounds"] == 1 and factcheck["swaps"] == 1
    # publication 1 + route 1 + extract 3 + outline 1 + write 4 + judge 4 + post 2
    # + flow overview 1 + details 2
    assert n_chats(model) == 19
    assert not model.loaded  # every model released at the end


def test_rerun_after_done_reuses_everything(model, arxiv, config):
    go(config, model, arxiv, auto_approve=True)
    calls = n_chats(model)
    go(config, model, arxiv)
    assert n_chats(model) == calls and arxiv.pdf_downloads == 1


def test_fresh_replay_reproduces_the_agentic_run_without_any_model(model, arxiv, config):
    first = go(config, model, arxiv, auto_approve=True, mode=ReplayMode.RECORD)
    slides = first.artifact("04_slides.json").read_text()
    claims = first.artifact("02_claims.json").read_text()
    again = run(config, ref=REF, mode=ReplayMode.REPLAY, fresh=True, http=arxiv.client())
    assert again.status == "done"
    assert again.artifact("04_slides.json").read_text() == slides
    assert again.artifact("02_claims.json").read_text() == claims
    spans = [s for s in read_trace(again.trace) if s["trace_id"] == again.trace_id]
    llm = [s for s in spans if s["name"] == "llm.chat"]
    assert len(llm) == 19 and all(s["cached"] for s in llm)
    assert (
        again.artifact("09_flows.json").read_text() == first.artifact("09_flows.json").read_text()
    )


def test_replay_without_cassettes_fails_loudly(arxiv, config):
    with pytest.raises(CassetteMissError):
        run(config, ref=REF, mode=ReplayMode.REPLAY, http=arxiv.client())


def test_local_pdf_run(model, config, tmp_path):
    pdf = make_pdf(tmp_path / "Own Paper.pdf")
    result = run(config, pdf=pdf, title="My Own Paper", auto_approve=True, client=model.client())
    assert result.run_dir.name == "own-paper" and result.overview.exists()


def test_run_fails_clearly_when_nothing_survives_the_fact_check(fake, arxiv, config):
    def liar(body):
        response = agentic_chat(body)
        props = body["format"]["properties"]
        if "bullets" in props and "verdicts" not in props:
            content = json.loads(response["message"]["content"])
            for b in content["bullets"]:
                b["text"] = "WRONG " + b["text"]
            response["message"]["content"] = json.dumps(content)
        return response

    fake.chat_handler = liar
    with pytest.raises(NothingSupportedError, match="05_factcheck.json"):
        run(config, ref=REF, auto_approve=True, client=fake.client(), http=arxiv.client())
    assert not fake.loaded


def test_the_main_figure_is_on_the_first_page(model, config, tmp_path):
    pdf = make_pdf(tmp_path / "Figs.pdf", figure=True)
    result = run(config, pdf=pdf, auto_approve=True, client=model.client())
    with pymupdf.open(result.overview) as doc:
        assert doc[0].get_images()  # the paper figure
        assert "synthetic architecture diagram" in doc[0].get_text()  # its caption


def test_only_the_two_pdfs_and_the_diagrams_are_written(model, arxiv, config):
    result = go(config, model, arxiv, auto_approve=True)
    outputs = {p.name for p in result.run_dir.iterdir() if p.suffix in (".pdf", ".png", ".md")}
    assert outputs == {
        "paper.pdf",
        "overview.pdf",
        "post.pdf",
        "flow.png",
        "flow-a.png",
        "flow-b.png",
    }
    assert not list(result.run_dir.glob("*.typ"))  # Typst sources are removed after rendering
    flows = Flows.model_validate_json(result.artifact("09_flows.json").read_text())
    assert [d.node_id for d in flows.details] == ["n1", "n2"]
