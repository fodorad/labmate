import csv
import json

import httpx
import pymupdf
import pytest

from labmate.cli import main
from labmate.config import load_config
from labmate.core.llm.client import OllamaClient
from labmate.paper2flow.chain import run_id_for
from tests.conftest import agentic_chat, make_pdf


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_lock_writes_digests(fake, workdir, capsys):
    assert main(["lock"], client=fake.client()) == 0
    lock = json.loads((workdir / "models.lock").read_text())
    assert lock["gemma4:26b-mlx"].startswith("21c59a2eae30")
    assert len(lock) == 3
    assert "CHANGED" not in capsys.readouterr().out


def test_lock_keeps_other_pinned_models(fake, workdir):
    (workdir / "models.lock").write_text('{"phi4-reasoning:plus": "abc"}')
    assert main(["lock"], client=fake.client()) == 0
    lock = json.loads((workdir / "models.lock").read_text())
    assert lock["phi4-reasoning:plus"] == "abc" and len(lock) == 4


def test_lock_flags_changed_digests(fake, workdir, capsys):
    main(["lock"], client=fake.client())
    fake.installed["qwen3.6:35b-mlx"] = "ffff" + "0" * 60
    assert main(["lock"], client=fake.client()) == 0
    out = capsys.readouterr().out
    assert "qwen3.6:35b-mlx" in out and "CHANGED" in out


def test_lock_fails_when_model_missing(fake, workdir, capsys):
    del fake.installed["gemma4:26b-mlx"]
    assert main(["lock"], client=fake.client()) == 1
    assert "gemma4:26b-mlx" in capsys.readouterr().err
    assert not (workdir / "models.lock").exists()


def test_probe_success_writes_report(fake, workdir):
    assert main(["probe", "--out", "p"], client=fake.client()) == 0
    report = json.loads((workdir / "p" / "probe_report.json").read_text())
    assert report["ollama_version"] == "0.24.0"
    assert {r["model"] for r in report["results"]} == {"qwen3.6:35b-mlx", "gemma4:26b-mlx"}


def test_probe_returns_1_when_a_check_fails(fake, workdir):
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": "nope"}}
    assert main(["probe", "--out", "p"], client=fake.client()) == 1


def test_unreachable_server_exits_2(workdir, capsys):
    def refuse(request):
        raise httpx.ConnectError("refused")

    client = OllamaClient(transport=httpx.MockTransport(refuse))
    assert main(["lock"], client=client) == 2
    assert "Cannot reach Ollama" in capsys.readouterr().err


def test_explicit_config_path(fake, workdir):
    (workdir / "alt.toml").write_text('[replay]\nlock_file = "pins.json"\n')
    assert load_config(workdir / "alt.toml").replay.lock_file.name == "pins.json"
    assert main(["--config", "alt.toml", "lock"], client=fake.client()) == 0
    assert (workdir / "pins.json").exists()


def test_paper2flow_and_paper2post_write_their_pdfs(fake, arxiv, workdir, capsys):
    fake.chat_handler = agentic_chat
    transport = arxiv.transport()
    assert main(["paper2flow", "2401.00001"], client=fake.client(), transport=transport) == 0
    assert "runs/2401.00001/overview.pdf" in capsys.readouterr().out
    assert main(["paper2post", "2401.00001"], client=fake.client(), transport=transport) == 0
    out = capsys.readouterr().out
    assert "runs/2401.00001/post.pdf" in out and "runs/2401.00001/trace.html" in out


def test_a_paper_from_a_url_gets_its_link_in_the_post(fake, workdir, capsys):
    pdf_bytes = make_pdf(workdir / "src.pdf").read_bytes()
    site = httpx.MockTransport(lambda r: httpx.Response(200, content=pdf_bytes))
    fake.chat_handler = agentic_chat
    url = "https://example.org/pdf/2023_My_Paper.pdf"
    assert main(["paper2post", url], client=fake.client(), transport=site) == 0
    with pymupdf.open(workdir / "runs" / run_id_for(url) / "post.pdf") as doc:
        assert url in " ".join(doc[0].get_text().split())


def test_replay_without_recordings_explains_what_to_do(fake, workdir, capsys):
    assert main(["paper2flow", "2401.00001", "--mode", "replay"], client=fake.client()) == 2
    err = capsys.readouterr().err
    assert "No cassette" in err and "run in auto mode" in err
    assert fake.paths().count("/api/chat") == 0


def test_a_missing_pdf_is_an_error(fake, workdir, capsys):
    assert main(["paper2flow", "missing.pdf"], client=fake.client()) == 1
    assert "error:" in capsys.readouterr().err


# --- evaluation commands ---------------------------------------------------------------


def _finish(fake, arxiv):
    fake.chat_handler = agentic_chat
    argv = ["paper2flow", "2401.00001"]
    assert main(argv, client=fake.client(), transport=arxiv.transport()) == 0


def test_eval_writes_results(fake, arxiv, workdir, capsys):
    _finish(fake, arxiv)
    (workdir / "runs" / "never-finished").mkdir()
    assert main(["eval"]) == 0
    assert "| A Test Paper |" in (workdir / "evals" / "results.md").read_text()
    results = json.loads((workdir / "evals" / "results.json").read_text())
    assert [r["paper_id"] for r in results] == ["2401.00001"]
    assert main(["eval", "runs/2401.00001", "--out", "e2"]) == 0
    assert (workdir / "e2" / "results.md").exists()


def test_eval_and_labels_without_runs(workdir, capsys):
    assert main(["eval"]) == 1
    assert main(["labels"]) == 1
    assert "no finished runs" in capsys.readouterr().err


def test_labels_then_judges(fake, arxiv, workdir, capsys):
    _finish(fake, arxiv)
    assert main(["judges"], client=fake.client()) == 1
    assert "no labelled bullets" in capsys.readouterr().err

    assert main(["labels", "-n", "3"]) == 0
    assert "Added" in capsys.readouterr().out
    path = workdir / "evals" / "labels.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["human"] = "s"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert main(["judges"], client=fake.client()) == 0
    out = capsys.readouterr().out
    assert "`gemma4:26b-mlx`" in out and "`qwen3.6:35b-mlx`" in out
    judged = json.loads((workdir / "evals" / "judges.json").read_text())
    assert [r["accuracy"] for r in judged] == [1.0, 1.0]
    assert (workdir / "evals" / "judges_trace.jsonl").exists()

    # the same verdicts come back from cassettes alone
    argv = ["judges", "--mode", "replay", "--models", "gemma4:26b-mlx"]
    assert main(argv, client=fake.client()) == 0
    assert len(json.loads((workdir / "evals" / "judges.json").read_text())) == 1


def test_trace_viewer_for_a_run(fake, arxiv, workdir, capsys):
    _finish(fake, arxiv)
    assert main(["trace", "arXiv:2401.00001", "--all"]) == 0
    assert main(["trace", "runs/2401.00001"]) == 0
    assert "runs/2401.00001/trace.html" in capsys.readouterr().out
    assert main(["trace", "no-such-paper"]) == 1
    assert "no trace.jsonl" in capsys.readouterr().err
