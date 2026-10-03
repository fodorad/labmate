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
    "PlannedCard",
    "Section",
    "SectionClaims",
    "VerdictLabel",
]

# --- the planned cards ---------------------------------------------------------------------


class PlannedCard(BaseModel):
    """One card of the overview and the claim cards it is built on."""

    purpose: str = Field(description="One of task, challenges, method, results.")
    claim_ids: list[str] = Field(description="Ids of the claim cards it is built on.")


class Outline(BaseModel):
    """The cards of the overview in order, each with the claims it is built on."""

    cards: list[PlannedCard] = Field(min_length=1, max_length=4)


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
        paper: The ingested paper.
        claims: Verified claim cards.
        outline: The planned cards: claim cards grouped by kind (set by the write step).
        written: The cards as the writer drafted them.
        checked: The cards after the fact-check loop, with its audit.
        flows: The flow diagrams.
    """

    source: str
    title: str | None = None
    paper: Paper | None = None
    claims: Claims | None = None
    outline: Outline | None = None
    written: Cards | None = None
    checked: FactChecked | None = None
    flows: Flows | None = None
