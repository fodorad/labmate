import json

import httpx
import pytest

from labmate.cli import configured_models, main
from labmate.config import Config, load_config
from labmate.core.llm.client import OllamaClient


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_configured_models_are_normalized_and_unique():
    config = Config()
    assert configured_models(config) == ["qwen3.6:35b-mlx", "gemma4:26b-mlx"]
    config.models.critic = "qwen3.6:35b-mlx"
    config.models.text = "qwen3.6:35b-mlx"
    assert configured_models(config) == ["qwen3.6:35b-mlx"]


def test_lock_writes_digests(fake, workdir, capsys):
    assert main(["lock"], client=fake.client()) == 0
    lock = json.loads((workdir / "models.lock").read_text())
    assert lock["gemma4:26b-mlx"].startswith("21c59a2eae30")
    assert len(lock) == 2
    assert "CHANGED" not in capsys.readouterr().out


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


def test_probe_silences_per_request_http_logs(fake, workdir):
    import logging

    main(["probe", "--out", "p"], client=fake.client())
    assert logging.getLogger("httpx").level == logging.WARNING


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


def test_main_builds_a_real_client_when_none_injected(workdir, monkeypatch):
    built = {}

    class Recorder(OllamaClient):
        def __init__(self, host, timeout_s):
            built["host"] = host
            super().__init__(
                host,
                timeout_s,
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"models": []})),
            )

    monkeypatch.setattr("labmate.cli.OllamaClient", Recorder)
    assert main(["lock"]) == 1  # nothing installed on this fake server
    assert built["host"] == "http://localhost:11434"


def test_run_pauses_then_approve_finishes(fake, arxiv, workdir, capsys):
    from tests.conftest import agentic_chat

    fake.chat_handler = agentic_chat
    assert main(["paper2flow", "run", "2401.00001"], client=fake.client(), http=arxiv.client()) == 0
    assert "Outline ready for review: runs/2401.00001/outline.yaml" in capsys.readouterr().out
    assert (
        main(
            ["paper2flow", "run", "2401.00001", "--approve"],
            client=fake.client(),
            http=arxiv.client(),
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "runs/2401.00001/overview.pdf" in out and "runs/2401.00001/post.pdf" in out


def test_run_without_paper_is_an_error(fake, workdir, capsys):
    assert main(["paper2flow", "run"], client=fake.client()) == 1
    assert "give an arXiv id" in capsys.readouterr().err


# --- M5: evaluation commands ---------------------------------------------------------------


def _finish(fake, arxiv):
    from tests.conftest import agentic_chat

    fake.chat_handler = agentic_chat
    argv = ["paper2flow", "run", "2401.00001", "--auto-approve"]
    assert main(argv, client=fake.client(), http=arxiv.client()) == 0


def test_eval_writes_results(fake, arxiv, workdir, capsys):
    _finish(fake, arxiv)
    (workdir / "runs" / "paused-only").mkdir()
    assert main(["paper2flow", "eval"]) == 0
    assert "| A Test Paper |" in (workdir / "evals" / "results.md").read_text()
    results = json.loads((workdir / "evals" / "results.json").read_text())
    assert [r["paper_id"] for r in results] == ["2401.00001"]
    assert main(["paper2flow", "eval", "runs/2401.00001", "--out", "e2"]) == 0
    assert (workdir / "e2" / "results.md").exists()


def test_eval_and_labels_without_runs(workdir, capsys):
    assert main(["paper2flow", "eval"]) == 1
    assert main(["paper2flow", "labels"]) == 1
    assert "no finished runs" in capsys.readouterr().err


def test_labels_then_judges(fake, arxiv, workdir, capsys):
    import csv

    _finish(fake, arxiv)
    assert main(["paper2flow", "judges"], client=fake.client()) == 1
    assert "no labelled bullets" in capsys.readouterr().err

    assert main(["paper2flow", "labels", "-n", "3"]) == 0
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

    assert main(["paper2flow", "judges"], client=fake.client()) == 0
    out = capsys.readouterr().out
    assert "`gemma4:26b-mlx`" in out and "`qwen3.6:35b-mlx`" in out
    judged = json.loads((workdir / "evals" / "judges.json").read_text())
    assert [r["accuracy"] for r in judged] == [1.0, 1.0]
    assert (workdir / "evals" / "judges_trace.jsonl").exists()

    # the same verdicts come back from cassettes alone
    argv = ["paper2flow", "judges", "--mode", "replay", "--models", "gemma4:26b-mlx"]
    assert main(argv, client=fake.client()) == 0
    assert len(json.loads((workdir / "evals" / "judges.json").read_text())) == 1


# --- M6: trace viewer and gallery ----------------------------------------------------------


def test_trace_publish_verify_site(fake, arxiv, workdir, capsys):
    _finish(fake, arxiv)
    assert (workdir / "runs" / "2401.00001" / "trace.html").exists()  # written by `run`
    assert main(["trace", "arXiv:2401.00001", "--all"]) == 0
    assert main(["trace", "runs/2401.00001"]) == 0
    assert main(["trace", "no-such-paper"]) == 1
    assert "no trace.jsonl" in capsys.readouterr().err

    assert main(["paper2flow", "publish", "2401.00001"], http=arxiv.client()) == 0
    assert "Published gallery/2401.00001: replays exactly" in capsys.readouterr().out
    assert main(["paper2flow", "publish", "missing"]) == 1

    assert main(["paper2flow", "verify"], http=arxiv.client()) == 0
    assert "OK   2401.00001: 9 identical" in capsys.readouterr().out
    (workdir / "gallery" / "2401.00001" / "04_slides.json").write_text("{}")
    assert main(["paper2flow", "verify", "2401.00001"], http=arxiv.client()) == 1
    assert "different: 04_slides.json" in capsys.readouterr().out

    assert main(["paper2flow", "site"]) == 0
    assert (workdir / "site" / "2401.00001" / "index.html").exists()
    assert "1 paper(s)" in capsys.readouterr().out


def test_verify_empty_gallery(workdir, capsys):
    assert main(["paper2flow", "verify"]) == 0
    assert "Nothing to verify" in capsys.readouterr().out


# --- M7: engine choice ---------------------------------------------------------------------


def test_run_with_the_langgraph_engine(fake, arxiv, workdir, capsys):
    from tests.conftest import agentic_chat

    fake.chat_handler = agentic_chat
    argv = ["paper2flow", "run", "2401.00001", "--engine", "langgraph"]
    assert main(argv, client=fake.client(), http=arxiv.client()) == 0
    assert "Outline ready for review" in capsys.readouterr().out
    assert main([*argv, "--approve"], client=fake.client(), http=arxiv.client()) == 0
    assert "overview.pdf" in capsys.readouterr().out


def test_run_publish_and_verify_a_pdf_from_a_url(fake, workdir, capsys):
    import httpx

    from tests.conftest import agentic_chat, make_pdf

    pdf_bytes = make_pdf(workdir / "src.pdf").read_bytes()
    downloads = []

    def serve(request):
        downloads.append(str(request.url))
        return httpx.Response(200, content=pdf_bytes)

    http = httpx.Client(transport=httpx.MockTransport(serve))
    fake.chat_handler = agentic_chat
    url = "https://example.org/pdf/2023_My_Paper.pdf"
    argv = ["paper2flow", "run", "--pdf", url, "--auto-approve"]
    assert main(argv, client=fake.client(), http=http) == 0
    run_dir = workdir / "runs" / "2023-my-paper"
    paper = json.loads((run_dir / "00_paper.json").read_text())
    assert paper["url"] == url and paper["title"] == "A Test Paper"
    import pymupdf

    with pymupdf.open(run_dir / "post.pdf") as doc:
        assert "https://example.org/pdf/2023_My_Paper.pdf" in doc[0].get_text()

    assert main(["trace", url]) == 0
    assert main(["paper2flow", "publish", url], http=http) == 0
    capsys.readouterr()
    assert not (workdir / "gallery" / "2023-my-paper" / "2023-my-paper.pdf").exists()
    assert main(["paper2flow", "verify"], http=http) == 0  # re-downloads the PDF from its URL
    assert "OK   2023-my-paper" in capsys.readouterr().out
    assert len(downloads) == 3  # run, publish's replay check, verify
