"""Shared data model: papers, claim cards and the fact-check audit trail.

Every labmate feature reads and writes these, serialised as JSON artifacts.
"""

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


class Figure(BaseModel):
    """A figure cropped from the paper PDF.

    Attributes:
        id: ``fig1``, ``fig2``, ... by figure number.
        number: Figure number as printed in the caption.
        caption: Caption text (whitespace-normalised, truncated).
        page: 1-based page number.
        path: PNG path relative to the paper's run directory.
    """

    id: str
    number: int
    caption: str
    page: int
    path: str


class Paper(BaseModel):
    """An ingested paper.

    Attributes:
        paper_id: arXiv id (``"1706.03762"``) or a slug for local PDFs.
        title: Paper title.
        authors: Author names.
        abstract: Abstract text.
        url: Canonical link shown in the outputs.
        date: Publication date as the source states it (e.g. ``2017-06-12`` or
            ``21 September 2023``); empty if unknown.
        year: Publication year, if known.
        venue: Journal or conference, if the paper states one.
        notes: Publication notes from arXiv (journal reference, comments).
        sections: Body sections, references excluded.
        figures: Figures cropped from the PDF.
    """

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    url: str = ""
    date: str = ""
    year: int | None = None
    venue: str = ""
    notes: str = ""
    sections: list[Section] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)


# --- claims ------------------------------------------------------------------------------

ClaimKind = Literal["task", "challenge", "contribution", "method", "result", "limitation"]
"""What a claim card is about."""


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
    """

    id: str
    claim: str
    evidence_quote: str
    kind: ClaimKind
    section: str
    page: int


class Claims(BaseModel):
    """All claim cards of a paper plus what the guard rejected."""

    cards: list[ClaimCard]
    rejected: list[ClaimDraft] = Field(default_factory=list)


# --- grounded text and its fact-check ------------------------------------------------------


class Bullet(BaseModel):
    """A bullet (or sentence) and the claims that support it."""

    text: str = Field(description="At most 25 words.")
    claim_ids: list[str] = Field(min_length=1)


class Card(BaseModel):
    """A titled card of bullets, every bullet grounded in claim cards."""

    title: str = Field(description="At most 8 words.")
    bullets: list[Bullet] = Field(min_length=1, max_length=4)


class Cards(BaseModel):
    """Grounded cards in order: the overview's four cards, a post's sentences, an answer."""

    cards: list[Card]


# --- fact-check loop -------------------------------------------------------------------------

VerdictLabel = Literal["supported", "partial", "unsupported"]
"""Judge verdict for one bullet. Only ``supported`` passes."""


class BulletVerdict(BaseModel):
    """The judge's verdict on one bullet."""

    bullet: int = Field(description="1-based bullet number.")
    verdict: VerdictLabel
    reason: str = Field(description="One short sentence naming what is or isn't supported.")


class CardVerdicts(BaseModel):
    """Judge output for one card: exactly one verdict per bullet."""

    verdicts: list[BulletVerdict]


class BulletCheck(BaseModel):
    """Everything known about one bullet in one round (the fact-check audit trail).

    Attributes:
        card: 1-based card number.
        bullet: 1-based bullet number within the card.
        text: Bullet text.
        claim_ids: Cited claims.
        problems: Deterministic check failures (e.g. a number not in the evidence).
        verdict: Judge verdict, if the judge ran.
        reason: Judge reason.
    """

    card: int
    bullet: int
    text: str
    claim_ids: list[str]
    problems: list[str] = Field(default_factory=list)
    verdict: VerdictLabel | None = None
    reason: str = ""

    @property
    def passed(self) -> bool:
        """True if no deterministic problem and the judge said ``supported``."""
        return not self.problems and self.verdict == "supported"


class FactCheckReport(BaseModel):
    """Per-round audit of the fact-check loop plus headline numbers.

    Attributes:
        rounds: Checks of every bullet, per round (round 0 = the writer's first draft).
        dropped: Bullets still failing after the last round, removed from the cards.
        dropped_cards: 1-based numbers of cards that lost all their bullets.
    """

    rounds: list[list[BulletCheck]]
    dropped: list[BulletCheck] = Field(default_factory=list)
    dropped_cards: list[int] = Field(default_factory=list)

    @property
    def failed_first(self) -> int:
        """Bullets failing in the first draft."""
        return sum(not c.passed for c in self.rounds[0]) if self.rounds else 0

    @property
    def total_first(self) -> int:
        """Bullets in the first draft."""
        return len(self.rounds[0]) if self.rounds else 0


class FactChecked(BaseModel):
    """Output of the fact-check step: the corrected cards and the audit trail."""

    cards: Cards
    report: FactCheckReport
