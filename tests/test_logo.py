import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/fodorad/labmate/main/"


def test_the_readme_shows_the_logo_by_an_absolute_url_so_pypi_can_render_it():
    readme = (ROOT / "README.md").read_text()

    sources = re.findall(r'<img[^>]*src="([^"]+)"', readme) + re.findall(
        r"!\[[^\]]*\]\(([^)]+)\)", readme
    )

    assert sources, "the README has no image"
    assert all(src.startswith("https://") for src in sources)  # a relative path breaks on PyPI
    logo = next(src for src in sources if "logo" in src)
    assert logo.startswith(RAW) and (ROOT / logo.removeprefix(RAW)).is_file()


def test_the_logo_files_are_valid_svg_with_a_title():
    for name in ("logo.svg", "favicon.svg"):
        root = ET.parse(ROOT / "docs" / "_static" / name).getroot()

        assert root.tag.endswith("svg") and root.attrib["viewBox"] == "0 0 96 96"
        assert any(el.tag.endswith("title") and el.text == "labmate" for el in root)
