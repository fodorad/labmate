import json

import httpx
import yaml

from labmate.bench.guard import Memory
from labmate.cli import main
from tests.cv2job.conftest import cv_chat

ENOUGH = Memory(32, lambda: 64.0, share=0.75)
"""A 32 GB Mac with plenty free, whatever machine runs the tests."""


def test_micro_measures_each_model_cold_and_a_rerun_resumes(fake, tmp_path):
    out = tmp_path / "bench"
    argv = ["bench", "micro", "--models", "gemma4:e4b", "--out", str(out)]

    assert main(argv, ollama=fake.transport(), memory=ENOUGH) == 0

    saved = json.loads((out / "micro.json").read_text())["gemma4:e4b"]
    assert saved["size_gb"] == 6.6 and saved["load_s"] == 2.0  # the fake reports a 2 s load
    assert "`gemma4:e4b`" in (out / "micro.md").read_text()
    assert fake.loaded == set()  # unloaded when it ended
    chats = fake.paths().count("/api/chat")
    assert main(argv, ollama=fake.transport(), memory=ENOUGH) == 0
    assert fake.paths().count("/api/chat") == chats  # saved, not measured again


def test_triage_scores_each_decider_against_the_labels(fake, tmp_path):
    golden = yaml.safe_load(open("evals/triage/golden.yaml"))
    labels = [a for p in golden["papers"] for a in p["expected"].values()]
    out = tmp_path / "bench"

    assert main(["bench", "triage", "--models", "clef-flash", "--out", str(out)],
                ollama=fake.transport(), memory=ENOUGH) == 0  # fmt: skip

    result = json.loads((out / "triage.json").read_text())["clef-flash"]
    assert result["kind"] == "decision" and result["size_gb"] == 2.0
    assert result["exact"] == labels.count("post")  # the fake scores every paper 2.0: post


def test_macro_runs_a_use_case_on_each_profile_that_fits_and_scores_it(fake, tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('[profiles.small]\ntext = "gemma4:e4b"\ncritic = "gemma4:e4b"\n')
    fake.chat_handler = cv_chat
    out = tmp_path / "bench"
    argv = ["--config", str(config), "bench", "macro", "--use-cases", "cv2job", "--out", str(out)]

    assert main(argv, ollama=fake.transport(), memory=ENOUGH) == 0

    cell = json.loads((out / "macro.json").read_text())["cv2job__small"]
    assert cell["error"] == "" and cell["calls"] > 0 and cell["quality"] > 0
    assert "### cv2job" in capsys.readouterr().out


def test_the_benchmark_stops_before_loading_models_that_do_not_fit(fake, tmp_path, capsys):
    argv = ["bench", "micro", "--models", "gemma4:e4b", "--out", str(tmp_path)]

    assert main(argv, ollama=fake.transport(), memory=Memory(32, lambda: 1.0, share=0.75)) == 1

    assert "/api/chat" not in fake.paths() and "only 1.0 GB are free" in capsys.readouterr().err


def test_the_report_is_written_from_the_saved_results(fake, tmp_path):
    page = tmp_path / "benchmarks.md"

    assert main(["bench", "report", "--out", "evals/bench", "--docs", str(page)],
                ollama=fake.transport()) == 0  # fmt: skip

    text = page.read_text()
    assert "Ollama 0.30.0" in text  # asked the server
    for section in ("## Single models", "### ask: graph", "### scout", "## Triage deciders"):
        assert section in text
    assert "| `clef-flash` | decision | 10.9 GB | 30/36 |" in text  # size saved when rated


def test_the_report_needs_no_running_ollama(tmp_path):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    page = tmp_path / "benchmarks.md"

    assert main(["bench", "report", "--out", "evals/bench", "--docs", str(page)],
                ollama=httpx.MockTransport(refuse)) == 0  # fmt: skip

    text = page.read_text()
    assert "Ollama unknown" in text and "| `clef-flash` | decision | 10.9 GB | 30/36 |" in text
