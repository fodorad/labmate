import json

from labmate.cli import main
from labmate.triage.run import run_triage

SCORES = {"Alpha": 3, "Bravo": 5, "Charlie": 4, "Delta": 2, "Echo": 1}


def test_the_budget_keeps_only_the_best_papers_for_a_deep_read(fake, arxiv_five, config, tmp_path):
    from tests.triage.conftest import decider

    fake.chat_handler = decider("deep", scores=SCORES)

    decisions = run_triage(config, "attention", "video", tmp_path / "out", budget=2,
                           http=arxiv_five.transport(), ollama=fake.transport())  # fmt: skip

    deep = [d.title for d in decisions if d.action == "deep"]
    assert deep == ["Bravo", "Charlie"]  # highest relevance, not first in the search results
    assert {d.action for d in decisions if d.title not in deep} == {"skip"}
    rows = [json.loads(line) for line in (tmp_path / "out" / "decisions.jsonl").open()]
    assert [r["action"] for r in rows].count("deep") == 2
    table = (tmp_path / "out" / "triage.md").read_text()
    assert "Bravo" in table and "Fits the interests." in table


def test_a_dry_run_makes_no_overview(fake, arxiv_five, config, tmp_path):
    from tests.triage.conftest import decider

    fake.chat_handler = decider("deep")

    run_triage(config, "attention", "video", tmp_path / "out", budget=1,
               http=arxiv_five.transport(), ollama=fake.transport())  # fmt: skip

    assert not (tmp_path / "runs").exists()
    assert arxiv_five.pdf_downloads == 0


def test_run_makes_an_overview_for_each_deep_paper(fake, arxiv_five, config, tmp_path):
    from tests.triage.conftest import decider

    fake.chat_handler = decider("deep", scores=SCORES)

    run_triage(config, "attention", "video", tmp_path / "out", budget=1, run=True,
               http=arxiv_five.transport(), ollama=fake.transport())  # fmt: skip

    assert (tmp_path / "runs" / "2401.00002" / "overview.pdf").exists()  # Bravo
    assert not (tmp_path / "runs" / "2401.00003").exists()


def test_the_decision_model_is_the_decider_role(fake, arxiv_five, config, tmp_path):
    from tests.conftest import INSTALLED
    from tests.triage.conftest import decider

    fake.chat_handler = decider("skip")
    config.models.decider = INSTALLED[1]

    run_triage(config, "attention", "video", tmp_path / "out",
               http=arxiv_five.transport(), ollama=fake.transport())  # fmt: skip

    models = {b["model"] for p, b in fake.requests if p == "/api/chat"}
    assert models == {INSTALLED[1]}


def test_the_command_reads_the_interests_from_a_file(
    fake, arxiv_five, tmp_path, monkeypatch, capsys
):
    from tests.triage.conftest import decider

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[cache]\npath = "{tmp_path / "cache.sqlite"}"\n')
    (tmp_path / "interests.md").write_text("video emotion recognition")
    fake.chat_handler = decider("skip")

    code = main(["triage", "attention", "--interests", str(tmp_path / "interests.md"),
                 "--out", str(tmp_path / "out")],
                web=arxiv_five.transport(), ollama=fake.transport())  # fmt: skip

    assert code == 0
    assert "video emotion recognition" in fake.requests[0][1]["messages"][-1]["content"]
    assert str(tmp_path / "out" / "triage.md") in capsys.readouterr().out
