import json

import httpx
import pytest

from paper2carousel.cli import configured_models, main
from paper2carousel.config import Config, load_config
from paper2carousel.llm.client import OllamaClient


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_configured_models_are_normalized_and_unique():
    config = Config()
    config.models.image = "x/flux2-klein"
    tags = configured_models(config)
    assert tags.count("x/flux2-klein:latest") == 1
    assert tags[:3] == ["qwen3.6:35b-mlx", "gemma4:26b-mlx", "gemma4:e4b"]


def test_lock_writes_digests(fake, workdir, capsys):
    assert main(["lock"], client=fake.client()) == 0
    lock = json.loads((workdir / "models.lock").read_text())
    assert lock["gemma4:26b-mlx"].startswith("21c59a2eae30")
    assert len(lock) == 5
    assert "CHANGED" not in capsys.readouterr().out


def test_lock_flags_changed_digests(fake, workdir, capsys):
    main(["lock"], client=fake.client())
    fake.installed["qwen3.6:35b-mlx"] = "ffff" + "0" * 60
    assert main(["lock"], client=fake.client()) == 0
    out = capsys.readouterr().out
    assert "qwen3.6:35b-mlx" in out and "CHANGED" in out


def test_lock_fails_when_model_missing(fake, workdir, capsys):
    del fake.installed["x/flux2-klein:latest"]
    assert main(["lock"], client=fake.client()) == 1
    assert "x/flux2-klein:latest" in capsys.readouterr().err
    assert not (workdir / "models.lock").exists()


def test_probe_success_writes_report(fake, workdir):
    assert main(["probe", "--out", "p", "--skip-images"], client=fake.client()) == 0
    report = json.loads((workdir / "p" / "probe_report.json").read_text())
    assert report["ollama_version"] == "0.24.0"
    assert {r["model"] for r in report["results"]} == {
        "qwen3.6:35b-mlx",
        "gemma4:26b-mlx",
        "gemma4:e4b",
    }


def test_probe_silences_per_request_http_logs(fake, workdir):
    import logging

    main(["probe", "--out", "p", "--skip-images"], client=fake.client())
    assert logging.getLogger("httpx").level == logging.WARNING


def test_probe_returns_1_when_a_check_fails(fake, workdir):
    fake.chat_handler = lambda b: {"model": b["model"], "message": {"content": "nope"}}
    assert main(["probe", "--out", "p", "--skip-images"], client=fake.client()) == 1


def test_probe_includes_image_candidates_by_default(fake, workdir):
    main(["probe", "--out", "p"], client=fake.client())
    models = {
        r["model"] for r in json.loads((workdir / "p/probe_report.json").read_text())["results"]
    }
    assert "x/z-image-turbo:latest" in models and "x/flux2-klein:latest" in models


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

    monkeypatch.setattr("paper2carousel.cli.OllamaClient", Recorder)
    assert main(["lock"]) == 1  # nothing installed on this fake server
    assert built["host"] == "http://localhost:11434"


def test_run_pauses_then_approve_finishes(fake, arxiv, workdir, capsys):
    from tests.conftest import agentic_chat

    fake.chat_handler = agentic_chat
    assert main(["run", "2401.00001"], client=fake.client(), http=arxiv.client()) == 0
    assert "Outline ready for review: runs/2401.00001/outline.yaml" in capsys.readouterr().out
    assert main(["run", "2401.00001", "--approve"], client=fake.client(), http=arxiv.client()) == 0
    assert "runs/2401.00001/carousel.pdf" in capsys.readouterr().out


def test_run_baseline_flag(fake, arxiv, workdir, capsys):
    from tests.conftest import deck_chat

    fake.chat_handler = deck_chat
    argv = ["run", "2401.00001", "--baseline"]
    assert main(argv, client=fake.client(), http=arxiv.client()) == 0
    assert "runs/2401.00001/baseline/carousel.pdf" in capsys.readouterr().out


def test_run_without_paper_is_an_error(fake, workdir, capsys):
    assert main(["run"], client=fake.client()) == 1
    assert "give an arXiv id" in capsys.readouterr().err


# --- M5: evaluation commands ---------------------------------------------------------------


def _finish(fake, arxiv):
    from tests.conftest import agentic_chat

    fake.chat_handler = agentic_chat
    argv = ["run", "2401.00001", "--auto-approve"]
    assert main(argv, client=fake.client(), http=arxiv.client()) == 0


def test_eval_writes_results(fake, arxiv, workdir, capsys):
    _finish(fake, arxiv)
    (workdir / "runs" / "paused-only").mkdir()
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
    import csv

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
