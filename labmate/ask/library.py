"""The library: which documents ask may answer from, and how much each one counts.

``library/library.toml`` lists the sources::

    [[source]]
    id = "dissertation"
    file = "dissertation.pdf"
    title = "Multimodal ... (PhD dissertation)"
    label = "Dissertation"
    tier = 1
    year = 2025

    [[source]]
    id = "blinklinmult"
    file = "papers/2023_Fodor_BlinkLinMulT.pdf"
    label = "BlinkLinMulT"
    tier = 2
    theses = ["II"]          # thesis points of the dissertation this paper backs

Tier 1 is the source of truth (the dissertation), tier 2 your papers, tier 3 outside
context (e.g. the paper2flow gallery). Retrieval starts at tier 1 and widens only when
the evidence is not enough.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal[1, 2, 3]
"""1 = source of truth, 2 = own papers, 3 = outside context."""

TIER_NAMES = {1: "dissertation", 2: "own papers", 3: "outside context"}
"""How tiers are described to the models and in the UI."""


class Source(BaseModel):
    """One document of the library.

    Attributes:
        id: Short stable identifier (used in chunk ids, e.g. ``dissertation:0042``).
        file: PDF path relative to the library folder.
        title: Full title.
        label: Short name used in citations ("Dissertation", "BlinkLinMulT").
        tier: 1, 2 or 3.
        year: Publication year.
        venue: Journal, conference or institution.
        theses: Thesis points of the dissertation this source backs (tier 2).
    """

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    file: str
    title: str = ""
    label: str = ""
    tier: Tier = 2
    year: int | None = None
    venue: str = ""
    theses: list[str] = Field(default_factory=list)

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
        LibraryError: If the manifest is missing, ids repeat, or no source is tier 1.
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
    if library.sources and not any(s.tier == 1 for s in library.sources):
        raise LibraryError("no tier-1 source: the dissertation is the source of truth")
    return library
