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
        url: Canonical link shown on the slides.
        sections: Body sections, references excluded.
    """

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    url: str = ""
    sections: list[Section] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)


class DraftSlide(BaseModel):
    """One slide of the walking-skeleton deck (also the render input)."""

    title: str = Field(description="Short slide title, at most 8 words.")
    bullets: list[str] = Field(
        description="1 to 3 bullets, each at most 20 words.", min_length=1, max_length=3
    )
    image: str | None = Field(default=None, exclude_if=lambda v: v is None)
    image_caption: str | None = Field(default=None, exclude_if=lambda v: v is None)


class Deck(BaseModel):
    """A carousel: an ordered list of slides."""

    title: str = Field(description="Hook headline for the cover slide, at most 10 words.")
    slides: list[DraftSlide] = Field(min_length=1, max_length=10)
    cover_image: str | None = Field(default=None, exclude_if=lambda v: v is None)


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
    claim_ids: list[str] = Field(
        description="Ids of the 2 to 4 claim cards this slide is built on."
    )


class Outline(BaseModel):
    """The orchestrator's plan for the carousel."""

    hook: str = Field(description="Cover headline that makes an ML engineer stop scrolling.")
    slides: list[OutlineSlide] = Field(min_length=3, max_length=8)


class Bullet(BaseModel):
    """A slide bullet and the claims that support it."""

    text: str = Field(description="At most 25 words.")
    claim_ids: list[str] = Field(min_length=1)


class SlideText(BaseModel):
    """Written slide content, every bullet grounded in claim cards."""

    title: str = Field(description="At most 8 words.")
    bullets: list[Bullet] = Field(min_length=1, max_length=3)


class WrittenSlides(BaseModel):
    """All written slides, in outline order."""

    hook: str
    slides: list[SlideText]

    def to_deck(self, visuals: list[Visual | None] | None = None) -> Deck:
        """Drop the citations and attach visuals for rendering.

        Args:
            visuals: One optional visual per slide.

        Returns:
            A render-ready deck.
        """
        visuals = visuals or [None] * len(self.slides)
        return Deck(
            title=self.hook,
            slides=[
                DraftSlide(
                    title=s.title,
                    bullets=[b.text for b in s.bullets],
                    image=v.path if v else None,
                    image_caption=v.caption if v else None,
                )
                for s, v in zip(self.slides, visuals, strict=True)
            ],
        )


# --- M3: fact-check loop ---------------------------------------------------------------------

VerdictLabel = Literal["supported", "partial", "unsupported"]
"""Judge verdict for one bullet. Only ``supported`` passes."""


class BulletVerdict(BaseModel):
    """The judge's verdict on one bullet."""

    bullet: int = Field(description="1-based bullet number.")
    verdict: VerdictLabel
    reason: str = Field(description="One short sentence naming what is or isn't supported.")


class SlideVerdicts(BaseModel):
    """Judge output for one slide: exactly one verdict per bullet."""

    verdicts: list[BulletVerdict]


class BulletCheck(BaseModel):
    """Everything known about one bullet in one round (the fact-check audit trail).

    Attributes:
        slide: 1-based slide number.
        bullet: 1-based bullet number within the slide.
        text: Bullet text.
        claim_ids: Cited claims.
        problems: Deterministic check failures (e.g. a number not in the evidence).
        verdict: Judge verdict, if the judge ran.
        reason: Judge reason.
    """

    slide: int
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
        dropped: Bullets still failing after the last round, removed from the deck.
        dropped_slides: 1-based numbers of slides that lost all their bullets.
    """

    rounds: list[list[BulletCheck]]
    dropped: list[BulletCheck] = Field(default_factory=list)
    dropped_slides: list[int] = Field(default_factory=list)

    @property
    def failed_first(self) -> int:
        """Bullets failing in the first draft."""
        return sum(not c.passed for c in self.rounds[0]) if self.rounds else 0

    @property
    def total_first(self) -> int:
        """Bullets in the first draft."""
        return len(self.rounds[0]) if self.rounds else 0


class FactChecked(BaseModel):
    """Output of the fact-check step: the corrected slides and the audit trail."""

    slides: WrittenSlides
    report: FactCheckReport


# --- M4: visuals ----------------------------------------------------------------------------


class Visual(BaseModel):
    """An image placed on a slide.

    Attributes:
        kind: A figure cropped from the paper, a diagram or a chart drawn by the agent.
        source: Figure id (``fig2``), ``diagram`` or ``chart``.
        path: Image path (PNG or SVG) relative to the paper's run directory.
        caption: Caption shown under the image.
    """

    kind: Literal["figure", "diagram", "chart"]
    source: str
    path: str
    caption: str


class AgentStep(BaseModel):
    """One tool call of the visuals agent, with what the tool answered."""

    slide: int
    tool: str
    arguments: dict[str, object]
    observation: str


class Visuals(BaseModel):
    """Visuals agent output: one optional visual per slide plus the full tool-call log."""

    slides: list[Visual | None]
    steps: list[AgentStep] = Field(default_factory=list)


class SlideReview(BaseModel):
    """The vision model's review of one rendered page."""

    legible: bool = Field(description="All text and the visual are readable at phone size.")
    overflow: bool = Field(description="Something is cut off, overlapping or outside the page.")
    visual_relevant: bool = Field(
        description="The image (if any) illustrates the slide's text. True if there is no image."
    )
    alt_text: str = Field(description="Alt text for screen readers, at most 40 words.")


class Review(BaseModel):
    """Slide critic output: one review per page (cover first) and the visuals it removed."""

    pages: list[SlideReview]
    dropped_visuals: list[int] = Field(default_factory=list)


class PostDraft(BaseModel):
    """A LinkedIn post drafted from the fact-checked slides."""

    hook: str = Field(description="First line, at most 15 words, makes people open the post.")
    takeaways: list[Bullet] = Field(
        description="3 to 5 sentences explaining the paper, each citing its claim ids.",
        min_length=3,
        max_length=5,
    )
    question: str = Field(description="A closing question to readers, with no factual claims.")


class Post(BaseModel):
    """The fact-checked post and its audit trail."""

    hook: str
    takeaways: list[Bullet]
    question: str
    report: FactCheckReport
