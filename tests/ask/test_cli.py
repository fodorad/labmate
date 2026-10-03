import shutil

import pytest

from labmate.cli import main


@pytest.fixture
def workdir(tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    toml = (
        f'[cache]\npath = "{config.cache.path}"\n'
        f'[ask]\nlibrary = "{config.ask.library}"\n'
        "chunk_words = 30\ntop_k = 4\n"
    )
    (tmp_path / "config.toml").write_text(toml)
    return tmp_path


def test_index_then_query_the_graph_and_the_agent(workdir, model, capsys):
    ollama = model.transport()
    assert main(["ask", "query", "anything"], ollama=ollama) == 1
    assert "index is empty" in capsys.readouterr().err

    assert main(["ask", "index"], ollama=ollama) == 0
    assert "Indexed 3 source(s)" in capsys.readouterr().out

    assert main(["ask", "query", "Which datasets are used for training?"], ollama=ollama) == 0
    assert "[1] Dissertation §" in capsys.readouterr().out

    argv = ["ask", "query", "How does the transformer fuse landmarks?", "--choice", "2"]
    assert main(argv, ollama=ollama) == 0

    assert main(["ask", "query", "CEW ZJU datasets", "--agent", "agent"], ollama=ollama) == 0
    assert "(dissertation:" in capsys.readouterr().out


def test_removed_sentences_are_reported(workdir, model, capsys):
    ollama = model.transport()
    main(["ask", "index"], ollama=ollama)
    capsys.readouterr()

    assert main(["ask", "query", "Which WRONG datasets are used?"], ollama=ollama) == 0

    assert "unsupported sentence(s) removed" in capsys.readouterr().out


def test_eval_answers_needs_a_golden_set_and_scores_both_agents(workdir, model, config, capsys):
    ollama = model.transport()
    main(["ask", "index"], ollama=ollama)
    assert main(["ask", "eval-answers"], ollama=ollama) == 1
    assert "golden.yaml" in capsys.readouterr().err

    (config.ask.library / "golden.yaml").write_text(
        "- question: Which datasets are used?\n  sources: [dissertation]\n"
    )
    assert main(["ask", "eval-answers"], ollama=ollama) == 0

    table = (workdir / "evals" / "ask" / "answers.md").read_text()
    assert "| graph |" in table and "| agent |" in table


def test_a_missing_library_manifest_is_an_error(workdir, model, config, capsys):
    (config.ask.library / "library.toml").unlink()

    assert main(["ask", "index"], ollama=model.transport()) == 1
    assert "library.toml" in capsys.readouterr().err


def test_a_private_library_outside_the_repo_is_used_in_place(
    workdir, model, library, tmp_path, capsys, monkeypatch
):
    private = tmp_path / "private-library"
    shutil.copytree(library, private)
    ollama = model.transport()

    assert main(["ask", "--library", str(private), "index"], ollama=ollama) == 0

    assert (private / "index.sqlite").exists()  # the index lives with the documents
    assert not (library / "index.sqlite").exists()  # the repo's own library stays untouched
    monkeypatch.setenv("LABMATE_LIBRARY", str(private))
    capsys.readouterr()
    assert main(["ask", "query", "Which datasets are used for training?"], ollama=ollama) == 0
    assert "[1] Dissertation §" in capsys.readouterr().out
