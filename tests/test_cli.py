import json

import httpx
import pymupdf
import pytest

from labmate.cli import main
from labmate.paper2flow.chain import run_id_for
from tests.conftest import agentic_chat, make_pdf


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_paper2flow_and_paper2post_write_their_pdfs(fake, arxiv, workdir, capsys):
    fake.chat_handler = agentic_chat
    web, ollama = arxiv.transport(), fake.transport()
    assert main(["paper2flow", "2401.00001"], web=web, ollama=ollama) == 0
    assert "runs/2401.00001/overview.pdf" in capsys.readouterr().out
    assert main(["paper2post", "2401.00001"], web=web, ollama=ollama) == 0
    assert "runs/2401.00001/post.pdf" in capsys.readouterr().out


def test_a_paper_from_a_url_gets_its_link_in_the_post(fake, workdir, capsys):
    pdf_bytes = make_pdf(workdir / "src.pdf").read_bytes()
    site = httpx.MockTransport(lambda r: httpx.Response(200, content=pdf_bytes))
    fake.chat_handler = agentic_chat
    url = "https://example.org/pdf/2023_My_Paper.pdf"
    assert main(["paper2post", url], web=site, ollama=fake.transport()) == 0
    with pymupdf.open(workdir / "runs" / run_id_for(url) / "post.pdf") as doc:
        assert url in " ".join(doc[0].get_text().split())


def test_an_unreachable_ollama_explains_what_to_do(workdir, arxiv, capsys):
    def refuse(request):
        raise httpx.ConnectError("refused")

    argv = ["paper2flow", "2401.00001"]
    assert main(argv, web=arxiv.transport(), ollama=httpx.MockTransport(refuse)) == 2
    assert "Ollama" in capsys.readouterr().err


# --- evaluation commands ---------------------------------------------------------------


def _finish(fake, arxiv):
    fake.chat_handler = agentic_chat
    argv = ["paper2flow", "2401.00001"]
    assert main(argv, web=arxiv.transport(), ollama=fake.transport()) == 0


def test_eval_writes_results(fake, arxiv, workdir, capsys):
    _finish(fake, arxiv)
    (workdir / "runs" / "never-finished").mkdir()
    assert main(["eval"]) == 0
    assert "| A Test Paper |" in (workdir / "evals" / "results.md").read_text()
    results = json.loads((workdir / "evals" / "results.json").read_text())
    assert [r["paper_id"] for r in results] == ["2401.00001"]
    assert main(["eval", "runs/2401.00001", "--out", "e2"]) == 0
    assert (workdir / "e2" / "results.md").exists()
