"""Pipeline data model. Every step reads and writes these, serialised as JSON artifacts."""

from __future__ import annotations

from typing import Literal

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


# --- M2: agentic pipeline --------------------------------------------------------------------

PaperType = Literal["method", "benchmark", "survey", "position"]
"""Narrative families; each has its own slide template (see ``steps/outline.py``)."""

ClaimKind = Literal["contribution", "method", "result", "limitation"]
"""What a claim card is about."""


class Route(BaseModel):
    """Routing decision: which narrative template fits the paper."""

    paper_type: PaperType
    confidence: float = Field(ge=0, le=1, description="0 to 1")
    reason: str = Field(description="One sentence.")


class ClaimDraft(BaseModel):
    """A claim as the extractor returns it, before it is checked against the paper."""

    claim: str = Field(description="One self-contained sentence in your own words.")
    evidence_quote: str = Field(
        description="A short verbatim quote (at most 30 words) copied from the section text."
    )
    kind: ClaimKind


class SectionClaims(BaseModel):
    """Extractor output for one section."""

    claims: list[ClaimDraft] = Field(max_length=6)


class ClaimCard(BaseModel):
    """A verified claim: its evidence quote was found in the paper.

    Attributes:
        id: Stable id (``c01``, ``c02``, ...) in reading order.
        claim: The claim, paraphrased.
        evidence_quote: Verbatim quote from the paper.
        kind: What the claim is about.
        section: Section title.
        page: Page where the section starts.
        match: Fuzzy-match score of the quote against the section text (0-100).
    """

    id: str
    claim: str
    evidence_quote: str
    kind: ClaimKind
    section: str
    page: int
    match: float


class Claims(BaseModel):
    """All claim cards of a paper plus what the guard rejected."""

    cards: list[ClaimCard]
    rejected: list[ClaimDraft] = Field(default_factory=list)


class OutlineSlide(BaseModel):
    """One planned slide."""

    title: str = Field(description="Working title, at most 8 words.")
    purpose: str = Field(description="One of the allowed purposes.")
    claim_ids: list[str] = Field(description="Ids of the claim cards this slide is built on.")


class Outline(BaseModel):
    """The orchestrator's plan for the carousel."""

    hook: str = Field(description="Cover headline that makes an ML engineer stop scrolling.")
    slides: list[OutlineSlide] = Field(min_length=3, max_length=8)


class Bullet(BaseModel):
    """A slide bullet and the claims that support it."""

    text: str = Field(description="At most 20 words.")
    claim_ids: list[str] = Field(min_length=1)


class SlideText(BaseModel):
    """Written slide content, every bullet grounded in claim cards."""

    title: str = Field(description="At most 8 words.")
    bullets: list[Bullet] = Field(min_length=1, max_length=3)


class WrittenSlides(BaseModel):
    """All written slides, in outline order."""

    hook: str
    slides: list[SlideText]

    def to_deck(self) -> Deck:
        """Drop the citations for rendering.

        Returns:
            A render-ready deck.
        """
        return Deck(
            title=self.hook,
            slides=[
                DraftSlide(title=s.title, bullets=[b.text for b in s.bullets]) for s in self.slides
            ],
        )
