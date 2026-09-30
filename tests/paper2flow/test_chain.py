import json

import httpx
import pymupdf
import pytest

from labmate.config import Config, ReplayMode
from labmate.core.llm.replay import CassetteMissError
from labmate.core.tracing import read_trace
from labmate.paper2flow.chain import (
    ARTIFACTS,
    NothingSupportedError,
    paper2flow,
    run_id_for,
    source_kind,
)
from labmate.paper2flow.schemas import Flows
from tests.conftest import agentic_chat, make_pdf

REF = "2401.00001"
CHATS = 17  # publication 1 + route 1 + extract 3 + outline 1 + write 4 + judge 4 + flows 1+2


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


def chats(fake):
    return fake.paths().count("/api/chat")


def test_papers_are_filed_under_their_arxiv_id_or_file_name():
    assert [source_kind(s) for s in ("1706.03762", "https://arxiv.org/abs/1706.03762v2")] == [
        "arxiv",
        "arxiv",
    ]
    assert source_kind("https://openreview.net/pdf?id=x") == "url"
    assert source_kind("papers/My Paper.pdf") == "pdf"
    assert run_id_for("arXiv:1706.03762v7") == "1706.03762"
    assert run_id_for("papers/My Paper.pdf") == "my-paper"
    assert run_id_for("https://x.org/a/My_Paper.pdf").startswith("my-paper-")


def test_a_paper_becomes_an_overview_pdf(model, arxiv, config):
    overview = paper2flow(config, REF, client=model.client(), transport=arxiv.transport())
    run_dir = overview.parent
    with pymupdf.open(overview) as doc:
        # cover, four cards, data flow, two detail flows
        assert doc.page_count == 5
        assert "A Test Paper" in doc[0].get_text()
        cards = " ".join(doc[1].get_text().split())
        assert all(k in cards for k in ["Task", "Challenges", "Proposed method", "Main results"])
        assert "END-TO-END DATA FLOW" in doc[2].get_text()
        assert "DETAIL A" in doc[3].get_text() and "DETAIL B" in doc[4].get_text()
        assert not doc.metadata["creationDate"]  # no timestamp: same run, same bytes
    assert all((run_dir / name).exists() for name in ARTIFACTS.values())
    assert {p.name for p in run_dir.glob("*.png")} == {"flow.png", "flow-a.png", "flow-b.png"}
    assert (run_dir / "trace.html").exists() and not list(run_dir.glob("*.typ"))
    flows = Flows.model_validate_json((run_dir / ARTIFACTS["flows"]).read_text())
    assert [d.node_id for d in flows.details] == ["n1", "n2"]
    assert chats(model) == CHATS and not model.loaded  # every model released at the end


def test_every_step_and_model_call_is_traced(model, arxiv, config):
    overview = paper2flow(config, REF, client=model.client(), transport=arxiv.transport())
    spans = read_trace(overview.parent / "trace.jsonl")
    by_id = {s["span_id"]: s for s in spans}
    (root,) = [s for s in spans if s["name"] == "run"]
    assert root["command"] == "paper2flow" and root["paper"] == REF and root["status"] == "ok"
    steps = [s["name"] for s in sorted(spans, key=lambda s: s["start_ts"]) if s["name"] != "run"
             and s["name"].startswith("step.")]  # fmt: skip
    assert steps == [
        "step.ingest", "step.publication", "step.route", "step.extract", "step.outline",
        "step.write", "step.factcheck", "step.flows", "step.render",
    ]  # fmt: skip
    calls = [s for s in spans if s["name"] == "llm.chat"]
    assert len(calls) == CHATS and all(by_id[c["parent_id"]]["name"].startswith("step.")
                                       for c in calls)  # fmt: skip
    extract = next(s for s in spans if s["name"] == "step.extract")
    assert sum(c["parent_id"] == extract["span_id"] for c in calls) == 3  # one per section
    factcheck = next(s for s in spans if s["name"] == "step.factcheck")
    assert factcheck["rounds"] == 1 and factcheck["failed_first"] == 0


def test_replay_reproduces_the_run_without_a_model_or_the_network(model, arxiv, config):
    first = paper2flow(config, REF, client=model.client(), transport=arxiv.transport())
    artifacts = {n: (first.parent / n).read_text() for n in ARTIFACTS.values()}
    again = paper2flow(config, REF, mode=ReplayMode.REPLAY)
    assert {n: (again.parent / n).read_text() for n in ARTIFACTS.values()} == artifacts
    spans = read_trace(again.parent / "trace.jsonl")
    latest = spans[-1]["trace_id"]
    replayed = [s for s in spans if s["trace_id"] == latest and s["name"] == "llm.chat"]
    assert len(replayed) == CHATS and all(s["cached"] for s in replayed)
    assert arxiv.pdf_downloads == 1


def test_replay_without_recordings_fails_loudly(config):
    with pytest.raises(CassetteMissError):
        paper2flow(config, REF, mode=ReplayMode.REPLAY)


def test_papers_from_a_pdf_url_or_a_local_file(model, config, tmp_path):
    pdf = make_pdf(tmp_path / "Own Paper.pdf")
    site = httpx.MockTransport(lambda r: httpx.Response(200, content=pdf.read_bytes()))
    url = "https://example.org/pdf/2023_My_Paper.pdf"
    from_url = paper2flow(config, url, client=model.client(), transport=site)
    paper = json.loads((from_url.parent / ARTIFACTS["paper"]).read_text())
    assert paper["url"] == url and from_url.parent.name == run_id_for(url)
    local = paper2flow(config, str(pdf), title="My Own Paper", client=model.client())
    assert local.parent.name == "own-paper"
    with pymupdf.open(local) as doc:
        assert "My Own Paper" in doc[0].get_text()


def test_the_main_figure_is_on_the_cover(model, config, tmp_path):
    pdf = make_pdf(tmp_path / "Figs.pdf", figure=True)
    with pymupdf.open(paper2flow(config, str(pdf), client=model.client())) as doc:
        assert doc[0].get_images()  # the paper figure
        assert "synthetic architecture diagram" in doc[0].get_text()  # its caption


def test_a_run_stops_clearly_when_nothing_survives_the_fact_check(fake, arxiv, config):
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
    with pytest.raises(NothingSupportedError, match=ARTIFACTS["checked"]):
        paper2flow(config, REF, client=fake.client(), transport=arxiv.transport())
    assert not fake.loaded
