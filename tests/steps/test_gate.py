import pytest

from paper2carousel.steps.gate import GateError, read_gate, write_gate
from tests.steps.test_outline import CLAIMS, FOUR, blocks


def test_roundtrip(tmp_path):
    o = blocks(["c01"], ["c02"], ["c03"], ["c04"])
    path = write_gate(tmp_path / "outline.yaml", o, CLAIMS, "1706.03762", FOUR)
    text = path.read_text()
    assert text.startswith("# paper2carousel outline for review.")
    assert "#   c04 [result] claim 4 (S, p. 4)" in text
    assert read_gate(path, CLAIMS, "method") == o


def test_missing_file(tmp_path):
    with pytest.raises(GateError, match="run without --approve first"):
        read_gate(tmp_path / "outline.yaml", CLAIMS, "method")


@pytest.mark.parametrize("content", ["hook: [unclosed", "hook: h\nslides: []\n", "just a string"])
def test_malformed_files_are_explained(tmp_path, content):
    (tmp_path / "outline.yaml").write_text(content)
    with pytest.raises(GateError, match="not a valid outline"):
        read_gate(tmp_path / "outline.yaml", CLAIMS, "method")
