"""paper2flow data model: routing, the outline, the post and the flow diagrams.

The shared types (papers, claim cards, the fact-check audit) live in
:mod:`labmate.core.schemas` and are re-exported here, so a step imports all it needs
from one place.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from labmate.core.schemas import (
    Bullet,
    BulletCheck,
    BulletVerdict,
    ClaimCard,
    ClaimDraft,
    ClaimKind,
    Claims,
    FactChecked,
    FactCheckReport,
    Figure,
    Paper,
    Section,
    SectionClaims,
    SlideText,
    SlideVerdicts,
    VerdictLabel,
    WrittenSlides,
)

__all__ = [
    "Bullet",
    "BulletCheck",
    "BulletVerdict",
    "ClaimCard",
    "ClaimDraft",
    "ClaimKind",
    "Claims",
    "FactCheckReport",
    "FactChecked",
    "Figure",
    "Paper",
    "Section",
    "SectionClaims",
    "SlideText",
    "SlideVerdicts",
    "VerdictLabel",
    "WrittenSlides",
]

# --- routing and the outline ---------------------------------------------------------------

PaperType = Literal["method", "benchmark", "survey", "position"]
"""Narrative families; each has its own slide template (see ``steps/outline.py``)."""


class Route(BaseModel):
    """Routing decision: which narrative template fits the paper."""

    paper_type: PaperType
    confidence: float = Field(ge=0, le=1, description="0 to 1")
    reason: str = Field(description="One sentence.")


class OutlineSlide(BaseModel):
    """One planned slide."""

    title: str = Field(description="Working title, at most 8 words.")
    purpose: str = Field(description="One of the allowed purposes.")
    claim_ids: list[str] = Field(
        description="Ids of the 2 to 4 claim cards this slide is built on."
    )


class Outline(BaseModel):
    """The orchestrator's plan: the four blocks and the claims each one is built on."""

    hook: str = Field(description="Headline that makes an ML engineer stop scrolling.")
    slides: list[OutlineSlide] = Field(min_length=3, max_length=8)


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


# --- publication info -------------------------------------------------------------------------


class PublicationDraft(BaseModel):
    """Venue and date as the model reads them from the paper's first page."""

    venue: str = Field(
        description="Journal or conference name exactly as written in the source, or ''."
    )
    date: str = Field(description="Publication date exactly as written in the source, or ''.")


# --- flow diagrams -------------------------------------------------------------------------

NodeKind = Literal["input", "data", "component", "process", "output"]
"""Role of a node in a flow diagram (drives its colour and shape)."""


class FlowNode(BaseModel):
    """A box in a flow diagram."""

    id: str = Field(description="Short identifier, e.g. 'rgb'.")
    label: str = Field(description="At most 5 words, named as in the paper.")
    kind: NodeKind


class FlowEdge(BaseModel):
    """An arrow in a flow diagram: what flows from one box to the next."""

    source: str = Field(description="Node id.")
    target: str = Field(description="Node id.")
    label: str = Field(default="", description="What flows along it, at most 4 words, or ''.")


class FlowGraph(BaseModel):
    """A top-to-bottom flow diagram."""

    title: str = Field(description="At most 8 words.")
    nodes: list[FlowNode] = Field(min_length=3, max_length=10)
    edges: list[FlowEdge] = Field(min_length=2, max_length=16)
    caption: str = Field(description="One or two sentences describing the flow, at most 40 words.")


class FlowOverview(FlowGraph):
    """The end-to-end data flow, from the raw data to the target output."""

    expand: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Ids of the 1 to 3 steps that deserve their own detail diagram.",
    )


class FlowDetail(BaseModel):
    """A detail diagram breaking down one step of the overview."""

    node_id: str
    graph: FlowGraph


class Flows(BaseModel):
    """All flow diagrams of a paper: the overview and its detail diagrams, in order."""

    overview: FlowOverview
    details: list[FlowDetail] = Field(default_factory=list)
