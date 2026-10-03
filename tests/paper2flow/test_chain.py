import json

import httpx
import pymupdf
import pytest

from labmate.config import Config
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
    cfg.cache.path = tmp_path / "cache" / "replies.sqlite"
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
    overview = paper2flow(config, REF, web=arxiv.transport(), ollama=model.transport())
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
    assert not list(run_dir.glob("*.typ"))
    flows = Flows.model_validate_json((run_dir / ARTIFACTS["flows"]).read_text())
    assert [d.node_id for d in flows.details] == ["n1", "n2"]
    assert chats(model) == CHATS


def test_a_rerun_is_answered_from_the_cache(model, arxiv, config):
    first = paper2flow(config, REF, web=arxiv.transport(), ollama=model.transport())
    artifacts = {n: (first.parent / n).read_text() for n in ARTIFACTS.values()}

    again = paper2flow(config, REF, web=arxiv.transport(), ollama=model.transport())

    assert chats(model) == CHATS  # no model call the second time
    assert {n: (again.parent / n).read_text() for n in ARTIFACTS.values()} == artifacts


def test_papers_from_a_pdf_url_or_a_local_file(model, config, tmp_path):
    pdf = make_pdf(tmp_path / "Own Paper.pdf")
    site = httpx.MockTransport(lambda r: httpx.Response(200, content=pdf.read_bytes()))
    url = "https://example.org/pdf/2023_My_Paper.pdf"
    from_url = paper2flow(config, url, ollama=model.transport(), web=site)
    paper = json.loads((from_url.parent / ARTIFACTS["paper"]).read_text())
    assert paper["url"] == url and from_url.parent.name == run_id_for(url)
    local = paper2flow(config, str(pdf), title="My Own Paper", ollama=model.transport())
    assert local.parent.name == "own-paper"
    with pymupdf.open(local) as doc:
        assert "My Own Paper" in doc[0].get_text()


def test_the_main_figure_is_on_the_cover(model, config, tmp_path):
    pdf = make_pdf(tmp_path / "Figs.pdf", figure=True)
    with pymupdf.open(paper2flow(config, str(pdf), ollama=model.transport())) as doc:
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
        paper2flow(config, REF, web=arxiv.transport(), ollama=fake.transport())
