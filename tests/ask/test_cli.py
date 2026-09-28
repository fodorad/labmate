import pytest

from labmate.cli import main


@pytest.fixture
def workdir(tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    toml = (
        f'[replay]\ndir = "{config.replay.dir}"\nlock_file = "{config.replay.lock_file}"\n'
        f'[ask]\nlibrary = "{config.ask.library}"\nindex = "{config.ask.index}"\n'
        f'threads = "{config.ask.threads}"\nchunk_words = 30\ntop_k = 4\n'
    )
    (tmp_path / "config.toml").write_text(toml)
    return tmp_path


def test_cli_index_query_and_evals(workdir, model, config, capsys):
    client = model.client()
    assert main(["ask", "query", "anything"], client=client) == 1
    assert "index is empty" in capsys.readouterr().err
    assert main(["ask", "index"], client=client) == 0
    assert "Indexed 3 source(s)" in capsys.readouterr().out
    assert main(["ask", "query", "Which datasets are used for training?"], client=client) == 0
    out = capsys.readouterr().out
    assert "[1] Dissertation §" in out
    argv = ["ask", "query", "How does the transformer fuse landmarks?", "--choice", "2"]
    assert main(argv, client=client) == 0
    assert main(["ask", "query", "CEW ZJU datasets", "--agent", "prebuilt"], client=client) == 0
    assert "(dissertation:" in capsys.readouterr().out
    assert main(["ask", "query", "Which WRONG datasets are used?"], client=client) == 0
    assert "removed by the fact-check" in capsys.readouterr().out
    assert main(["ask", "eval-retrieval", "-n", "4", "--no-rewrite"], client=client) == 0
    assert (workdir / "evals" / "ask" / "retrieval.md").exists()
    assert main(["ask", "eval-answers"], client=client) == 1
    assert "golden.yaml" in capsys.readouterr().err
    (config.ask.library / "golden.yaml").write_text(
        "- question: Which datasets are used?\n  sources: [dissertation]\n"
    )
    assert main(["ask", "eval-answers"], client=client) == 0
    assert "| graph |" in (workdir / "evals" / "ask" / "answers.md").read_text()


def test_cli_conflict_note_and_missing_library(workdir, model, config, capsys, tmp_path):
    client = model.client()
    main(["ask", "index"], client=client)
    capsys.readouterr()
    question = "What F1 does BlinkLinMulT rarely reach on RT-BENE?"  # widens to the papers
    assert main(["ask", "query", question], client=client) == 0
    assert "note: F1 on RT-BENE: dissertation 0.912, other 0.905" in capsys.readouterr().out
    (config.ask.library / "library.toml").unlink()
    assert main(["ask", "index"], client=client) == 1
    assert "library.toml" in capsys.readouterr().err


def test_cli_unreachable_ollama(workdir, capsys):
    import httpx

    from labmate.core.llm.client import OllamaClient

    def refuse(request):
        raise httpx.ConnectError("refused")

    client = OllamaClient(transport=httpx.MockTransport(refuse))
    assert main(["ask", "index", "--fresh", "--mode", "live"], client=client) == 2
    assert "Cannot reach Ollama" in capsys.readouterr().err
