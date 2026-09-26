from pathlib import Path

import pymupdf
import pytest
import yaml

from paper2carousel.config import Config, ReplayMode
from paper2carousel.engines.plain import run, run_id_for
from paper2carousel.llm.replay import CassetteMissError
from paper2carousel.schemas import Claims, Deck, Outline, WrittenSlides
from paper2carousel.steps.gate import GateError
from paper2carousel.tracing import read_trace
from tests.conftest import DECK, agentic_chat, deck_chat, make_pdf

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
    assert not result.carousel.exists()
    text = result.gate.read_text()
    assert "make approve ARXIV=2401.00001" in text and "# Available claim cards:" in text
    outline = Outline.model_validate(yaml.safe_load(text))
    assert outline.slides[-1].purpose == "takeaway"
    claims = Claims.model_validate_json(result.artifact("02_claims.json").read_text())
    assert len(claims.cards) == 6 and all(c.match >= 90 for c in claims.cards)
    assert "qwen3.6:35b-mlx" in model.loaded  # not unloaded: the run is paused


def test_approve_finishes_and_respects_human_edits(model, arxiv, config):
    first = go(config, model, arxiv)
    edited = yaml.safe_load(first.gate.read_text())
    edited["hook"] = "Edited by a human"
    del edited["slides"][1]
    first.gate.write_text(yaml.safe_dump(edited))

    result = go(config, model, arxiv, approve=True)
    assert result.status == "done"
    written = WrittenSlides.model_validate_json(result.artifact("04_slides.json").read_text())
    assert written.hook == "Edited by a human" and len(written.slides) == 3
    with pymupdf.open(result.carousel) as doc:
        assert doc.page_count == 4
        assert "Edited by a human" in " ".join(doc[0].get_text().split())
    gate_span = next(
        s
        for s in read_trace(result.trace)
        if s["name"] == "step.gate" and s.get("status") == "approved"
    )
    assert gate_span["edited"] is True
    assert "qwen3.6:35b-mlx" not in model.loaded


def test_invalid_edit_is_rejected_with_the_rule_it_breaks(model, arxiv, config):
    first = go(config, model, arxiv)
    edited = yaml.safe_load(first.gate.read_text())
    edited["slides"][0]["claim_ids"] = ["c99"]
    first.gate.write_text(yaml.safe_dump(edited))
    with pytest.raises(GateError, match=r"unknown claim ids \['c99'\]"):
        go(config, model, arxiv, approve=True)


def test_auto_approve_runs_straight_through_and_traces_every_step(model, arxiv, config):
    result = go(config, model, arxiv, auto_approve=True)
    assert result.status == "done" and result.carousel.exists()
    spans = read_trace(result.trace)
    names = {s["name"] for s in spans}
    assert {"step.route", "step.extract", "step.outline", "step.gate", "step.write"} <= names
    extract = next(s for s in spans if s["name"] == "step.extract")
    llm_in_extract = [
        s for s in spans if s["name"] == "llm.chat" and s["parent_id"] == extract["span_id"]
    ]
    assert len(llm_in_extract) == 3  # one per section, parent kept across worker threads
    # route 1 + extract 3 + outline 1 + write 4
    assert n_chats(model) == 9


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
    assert len(llm) == 9 and all(s["cached"] for s in llm)


def test_replay_without_cassettes_fails_loudly(arxiv, config):
    with pytest.raises(CassetteMissError):
        run(config, ref=REF, mode=ReplayMode.REPLAY, http=arxiv.client())


def test_local_pdf_run(model, config, tmp_path):
    pdf = make_pdf(tmp_path / "Own Paper.pdf")
    result = run(config, pdf=pdf, title="My Own Paper", auto_approve=True, client=model.client())
    assert result.run_dir.name == "own-paper" and result.carousel.exists()


# --- baseline (M1 one-shot) ----------------------------------------------------------------


def test_baseline_runs_in_its_own_directory_and_shares_the_paper(fake, arxiv, config):
    fake.chat_handler = deck_chat
    result = run(config, ref=REF, baseline=True, client=fake.client(), http=arxiv.client())
    assert result.run_dir.name == "baseline" and result.status == "done"
    assert Deck.model_validate_json(result.artifact("01_deck.json").read_text()) == DECK
    assert (result.run_dir.parent / "00_paper.json").exists()
    with pymupdf.open(result.carousel) as doc:
        assert doc.page_count == len(DECK.slides) + 1
