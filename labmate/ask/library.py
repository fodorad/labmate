"""The library: which documents ask may answer from.

``library.toml`` in the library folder lists the sources::

    [[source]]
    id = "dissertation"
    file = "pdf/dissertation.pdf"
    title = "Multimodal ... (PhD dissertation)"
    label = "Dissertation"
    year = 2025
    venue = "ELTE"

``label`` is the short name used in citations ("Dissertation §4.2, p. 57").
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class Source(BaseModel):
    """One document of the library.

    Attributes:
        id: Short stable identifier (used in chunk ids, e.g. ``dissertation:0042``).
        file: PDF path relative to the library folder.
        title: Full title.
        label: Short name used in citations ("Dissertation", "BlinkLinMulT").
        year: Publication year.
        venue: Journal, conference or institution.
    """

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    file: str
    title: str = ""
    label: str = ""
    year: int | None = None
    venue: str = ""

    @property
    def name(self) -> str:
        """Citation label, falling back to the title and then the id."""
        return self.label or self.title or self.id


class Library(BaseModel):
    """All sources, and where their files are.

    Attributes:
        root: The library folder.
        sources: The documents, in manifest order.
    """

    root: Path
    sources: list[Source] = Field(default_factory=list)

    def get(self, source_id: str) -> Source:
        """Look up a source by id.

        Args:
            source_id: Source id.

        Returns:
            The source.

        Raises:
            KeyError: If there is no such source.
        """
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(source_id)

    def path(self, source: Source) -> Path:
        """Absolute path of a source's PDF.

        Args:
            source: A source of this library.

        Returns:
            The file path.
        """
        return self.root / source.file


class LibraryError(ValueError):
    """Raised when ``library.toml`` is missing or inconsistent."""


def load_library(root: Path) -> Library:
    """Read and validate ``<root>/library.toml``.

    Args:
        root: The library folder.

    Returns:
        The library.

    Raises:
        LibraryError: If the manifest is missing or ids repeat.
    """
    manifest = root / "library.toml"
    if not manifest.exists():
        raise LibraryError(f"no {manifest}; see labmate/ask/library.py for the format")
    with manifest.open("rb") as f:
        data = tomllib.load(f)
    library = Library(root=root, sources=[Source.model_validate(s) for s in data.get("source", [])])
    ids = [s.id for s in library.sources]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise LibraryError(f"duplicate source ids: {duplicates}")
    return library
