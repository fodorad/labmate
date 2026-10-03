from labmate.cli import main
from labmate.scout.agent import run_scout
from tests.scout.conftest import scripted, tool_results

NOTES_BOTH = "Two ways to cut attention cost: [2401.00001] and [2401.00002]."
NOTES_ONE = "Linear attention keeps accuracy while cutting memory [2401.00001]."


def test_the_notes_may_only_cite_papers_the_agent_has_read(fake, arxiv_two, config, tmp_path):
    fake.chat_handler = scripted(
        ("search_arxiv", {"query": "attention"}),
        ("write_notes", {"notes": NOTES_BOTH}),  # it only searched: not allowed
        ("read_abstract", {"arxiv_id": "2401.00001"}),
        ("write_notes", {"notes": NOTES_ONE}),
    )

    notes = run_scout(config, "attention", tmp_path / "out", http=arxiv_two.transport(),
                      ollama=fake.transport())  # fmt: skip

    assert notes is not None and notes.read_text().strip() == NOTES_ONE
    results = tool_results(fake)
    assert "you have not read" in results[1] and "2401.00001" in results[0]  # search hit listed
    assert results[-1] == "saved"


def test_the_deep_look_runs_paper2flow_and_is_limited(fake, arxiv_two, config, tmp_path):
    fake.chat_handler = scripted(
        ("read_overview", {"arxiv_id": "2401.00001"}),
        ("read_overview", {"arxiv_id": "2401.00002"}),
    )

    run_scout(config, "attention", tmp_path / "out", max_deep=1, http=arxiv_two.transport(),
              ollama=fake.transport())  # fmt: skip

    first, second = tool_results(fake)
    assert "A written card" in first and "overview:" in first  # the cards, then the PDF's path
    assert (tmp_path / "runs" / "2401.00001" / "overview.pdf").exists()
    assert "limit reached" in second
    assert not (tmp_path / "runs" / "2401.00002").exists()  # the second paper was never processed


def test_an_agent_that_stops_without_notes_is_an_error(
    fake, arxiv_two, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'[cache]\npath = "{tmp_path / "cache.sqlite"}"\n')
    fake.chat_handler = scripted(("search_arxiv", {"query": "attention"}))

    code = main(["scout", "attention", "--out", str(tmp_path / "out")],
                web=arxiv_two.transport(), ollama=fake.transport())  # fmt: skip

    assert code == 1 and "without writing notes" in capsys.readouterr().err
