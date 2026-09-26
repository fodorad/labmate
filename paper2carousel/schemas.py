"""Pipeline data model. Every step reads and writes these, serialised as JSON artifacts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A contiguous chunk of paper text.

    Attributes:
        title: Heading as it appears in the paper (or ``"Page N"`` for page fallback).
        page: 1-based page where the section starts.
        text: Plain text of the section.
    """

    title: str
    page: int
    text: str


class Paper(BaseModel):
    """An ingested paper.

    Attributes:
        paper_id: arXiv id (``"1706.03762"``) or a slug for local PDFs.
        title: Paper title.
        authors: Author names.
        abstract: Abstract text.
        url: Canonical link shown on the slides.
        sections: Body sections, references excluded.
    """

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    url: str = ""
    sections: list[Section] = Field(default_factory=list)


class DraftSlide(BaseModel):
    """One slide of the walking-skeleton deck."""

    title: str = Field(description="Short slide title, at most 8 words.")
    bullets: list[str] = Field(
        description="1 to 3 bullets, each at most 20 words.", min_length=1, max_length=3
    )


class Deck(BaseModel):
    """A carousel: an ordered list of slides."""

    title: str = Field(description="Hook headline for the cover slide, at most 10 words.")
    slides: list[DraftSlide] = Field(min_length=3, max_length=10)
