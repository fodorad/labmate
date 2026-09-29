import pytest

from labmate.ask.library import LibraryError, Source, load_library


def test_library_manifest(library):
    lib = load_library(library)
    assert [s.id for s in lib.sources] == ["dissertation", "blinklinmult", "outside"]
    assert lib.get("blinklinmult").theses == ["I"]
    assert lib.path(lib.get("dissertation")) == library / "dissertation.pdf"
    assert Source(id="x", file="x.pdf").name == "x"
    with pytest.raises(KeyError):
        lib.get("nope")


@pytest.mark.parametrize(
    ("toml", "error"),
    [
        (None, "no .*library.toml"),
        ('[[source]]\nid = "a"\nfile = "a.pdf"\ntier = 1\n[[source]]\nid = "a"\nfile = "b.pdf"\n',
         "duplicate source ids"),
        ('[[source]]\nid = "a"\nfile = "a.pdf"\ntier = 2\n', "no tier-1 source"),
    ],
)  # fmt: skip
def test_library_errors(tmp_path, toml, error):
    if toml is not None:
        (tmp_path / "library.toml").write_text(toml)
    with pytest.raises(LibraryError, match=error):
        load_library(tmp_path)
