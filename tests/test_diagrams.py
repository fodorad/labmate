from labmate.cli import main


def test_graphs_command_draws_every_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["graphs", "--out", "docs/graphs.md"]) == 0
    text = (tmp_path / "docs" / "graphs.md").read_text()
    assert text.count("```mermaid") == 7 and "research_retrieve" in text
