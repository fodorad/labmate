import json

import pymupdf
import pytest

from paper2carousel.config import Config, ReplayMode
from paper2carousel.engines.plain import run, run_id_for
from paper2carousel.llm.replay import CassetteMissError
from paper2carousel.schemas import Deck
from paper2carousel.tracing import read_trace
from tests.conftest import DECK, deck_chat, make_pdf


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.replay.dir = tmp_path / "cassettes"
    cfg.replay.lock_file = tmp_path / "models.lock"
    cfg.tracing.runs_dir = tmp_path / "runs"
    return cfg


def n_chats(fake):
    return fake.paths().count("/api/chat")


def test_run_id():
    assert run_id_for("arXiv:1706.03762v7", None) == "1706.03762"
    assert run_id_for(None, __import__("pathlib").Path("My Paper.pdf")) == "my-paper"
    with pytest.raises(ValueError):
        run_id_for(None, None)


def test_full_run_produces_artifacts_and_trace(fake, arxiv, config):
    fake.chat_handler = deck_chat
    result = run(config, ref="2401.00001", client=fake.client(), http=arxiv.client())
    assert Deck.model_validate_json(result.deck.read_text()) == DECK
    with pymupdf.open(result.carousel) as doc:
        assert doc.page_count == len(DECK.slides) + 1
    spans = {s["name"]: s for s in read_trace(result.trace)}
    assert {"run", "step.ingest", "step.draft", "step.render", "llm.chat"} <= spans.keys()
    assert spans["step.ingest"]["sections"] == 3 and spans["step.draft"]["slides"] == 3
    assert spans["llm.chat"]["parent_id"] == spans["step.draft"]["span_id"]
    assert "qwen3.6:35b-mlx" not in fake.loaded  # text model unloaded at the end


def test_rerun_reuses_checkpoints(fake, arxiv, config):
    fake.chat_handler = deck_chat
    run(config, ref="2401.00001", client=fake.client(), http=arxiv.client())
    run(config, ref="2401.00001", client=fake.client(), http=arxiv.client())
    assert n_chats(fake) == 1 and arxiv.pdf_downloads == 1


def test_fresh_replay_reproduces_without_any_model(fake, arxiv, config):
    fake.chat_handler = deck_chat
    first = run(
        config, ref="2401.00001", mode=ReplayMode.RECORD, client=fake.client(), http=arxiv.client()
    )
    recorded = first.deck.read_text()
    first.deck.unlink()
    again = run(config, ref="2401.00001", mode=ReplayMode.REPLAY, fresh=True, http=arxiv.client())
    assert again.deck.read_text() == recorded
    cached = [s for s in read_trace(again.trace) if s["name"] == "llm.chat"]
    assert cached[-1]["cached"] is True


def test_replay_without_cassettes_fails_loudly(arxiv, config):
    with pytest.raises(CassetteMissError):
        run(config, ref="2401.00001", mode=ReplayMode.REPLAY, http=arxiv.client())


def test_local_pdf_run(fake, config, tmp_path):
    fake.chat_handler = deck_chat
    pdf = make_pdf(tmp_path / "Own Paper.pdf")
    result = run(config, pdf=pdf, title="My Own Paper", client=fake.client())
    assert result.run_dir.name == "own-paper"
    assert json.loads(result.paper.read_text())["title"] == "My Own Paper"
    assert result.carousel.exists()
