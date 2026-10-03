from labmate.cli import main


def test_graphs_command_draws_every_chain_and_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["graphs", "--out", "docs/graphs.md"]) == 0
    text = (tmp_path / "docs" / "graphs.md").read_text()
    assert text.count("```mermaid") == 5
    assert all(
        f"## {name}" in text for name in ("paper2flow", "paper2post", "cv2job", "ask (graph)")
    )
    assert "grade" in text and "clarify" in text  # the ask graph's nodes
