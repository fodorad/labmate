import re
from pathlib import Path

from labmate.cli import main
from labmate.diagrams import pipeline_steps

ROOT = Path(__file__).parent.parent


def mermaid_blocks(path: Path) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", path.read_text(), re.DOTALL)


def test_graphs_command_draws_every_chain_and_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["graphs", "--out", "docs/graphs.md"]) == 0
    text = (tmp_path / "docs" / "graphs.md").read_text()
    assert text.count("```mermaid") == 5
    assert all(
        f"## {name}" in text for name in ("paper2flow", "paper2post", "cv2job", "ask (graph)")
    )
    assert "grade" in text and "clarify" in text  # the ask graph's nodes


def test_the_readme_shows_the_same_pipeline_diagrams_as_the_docs_page():
    page = mermaid_blocks(ROOT / "docs" / "pipelines.md")
    readme = (ROOT / "README.md").read_text()
    index = (ROOT / "docs" / "index.md").read_text()

    assert len(page) == 7  # one per use case; ask has its index, graph and agent
    assert all(block in readme and block in index for block in page)


def test_the_readme_gives_every_use_case_the_same_type_as_the_docs_page():
    page = (ROOT / "docs" / "pipelines.md").read_text()
    types = re.findall(r"^## (.+)\n\nType: (.+)\n", page, re.MULTILINE)

    assert len(types) == 7
    for name in ("README.md", "docs/index.md"):
        text = (ROOT / name).read_text()
        for use_case, kind in types:
            assert re.search(
                rf"^#+ {re.escape(use_case)}\n\nType: {re.escape(kind)}\n", text, re.M
            ), f"{name}: '{use_case}' should be followed by 'Type: {kind}'"


def test_every_step_of_a_chain_is_in_its_diagram():
    page = mermaid_blocks(ROOT / "docs" / "pipelines.md")
    diagrams = {
        "paper2flow (chain)": page[0],
        "paper2post (chain)": page[1],
        "cv2job (chain with one agent step)": page[3],
        "ask (graph)": page[5],
    }

    for title, steps in pipeline_steps().items():
        if title in diagrams:
            missing = [n for n in steps if not re.search(rf"\b{n}\b", diagrams[title])]
            assert not missing, f"{title}: steps {missing} are not in its diagram"
