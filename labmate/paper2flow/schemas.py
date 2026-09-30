"""paper2flow data model: routing, the outline, the flow diagrams and the chain's state.

The shared types (papers, claim cards, grounded cards, the fact-check audit) live in
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
    Card,
    Cards,
    CardVerdicts,
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
    VerdictLabel,
)

__all__ = [
    "Analysis",
    "Bullet",
    "BulletCheck",
    "BulletVerdict",
    "Card",
    "CardVerdicts",
    "Cards",
    "ClaimCard",
    "ClaimDraft",
    "ClaimKind",
    "Claims",
    "FactCheckReport",
    "FactChecked",
    "Figure",
    "FlowDetail",
    "FlowEdge",
    "FlowGraph",
    "FlowNode",
    "FlowOverview",
    "Flows",
    "NodeKind",
    "Outline",
    "Paper",
    "PaperType",
    "PlannedCard",
    "PublicationDraft",
    "Route",
    "Section",
    "SectionClaims",
    "VerdictLabel",
]

# --- routing and the outline ---------------------------------------------------------------

PaperType = Literal["method", "benchmark", "survey", "position"]
"""What kind of paper it is; it changes how the four cards and the flow are read."""


class Route(BaseModel):
    """Routing decision: which kind of paper this is."""

    paper_type: PaperType
    confidence: float = Field(ge=0, le=1, description="0 to 1")
    reason: str = Field(description="One sentence.")


class PlannedCard(BaseModel):
    """One planned card of the overview."""

    title: str = Field(description="Working title, at most 8 words.")
    purpose: str = Field(description="One of the allowed purposes.")
    claim_ids: list[str] = Field(description="Ids of the 3 to 5 claim cards it is built on.")


class Outline(BaseModel):
    """The orchestrator's plan: the four cards and the claims each one is built on."""

    cards: list[PlannedCard] = Field(min_length=4, max_length=4)


# --- publication info ----------------------------------------------------------------------


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
    """A top-to-bottom flow diagram, proposed by the model as data and checked by code."""

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


# --- the chain's state ---------------------------------------------------------------------


class Analysis(BaseModel):
    """What the analysis chain knows about a paper so far; each step fills in one field.

    Attributes:
        source: The paper as given: an arXiv id or URL, a PDF URL, or a local PDF path.
        title: Title override for PDFs without a usable title.
        paper: The ingested paper (venue and date added by the publication step).
        route: What kind of paper it is.
        claims: Verified claim cards.
        outline: The four planned cards.
        written: The cards as the writer drafted them.
        checked: The cards after the fact-check loop, with its audit.
        flows: The flow diagrams.
    """

    source: str
    title: str | None = None
    paper: Paper | None = None
    route: Route | None = None
    claims: Claims | None = None
    outline: Outline | None = None
    written: Cards | None = None
    checked: FactChecked | None = None
    flows: Flows | None = None
