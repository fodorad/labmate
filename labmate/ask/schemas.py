"""Data model of the ask agent: what the models return and what the graph carries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["thesis", "own_work", "related", "off_topic"]
"""What a question is about: a thesis point, the author's own work, related work, or
something the library can't answer."""


class Understanding(BaseModel):
    """The question, made self-contained, and what it is about."""

    standalone: str = Field(
        description="The question rewritten to stand alone (resolve 'it', 'that paper' "
        "from the conversation); the same question if it already stands alone."
    )
    intent: Intent
    options: list[str] = Field(
        default_factory=list,
        description="Only if the question could mean clearly different things in this "
        "library: 2 to 4 short readings to choose from. Otherwise empty.",
    )


class Plan(BaseModel):
    """Search queries that together cover the question."""

    queries: list[str] = Field(
        min_length=1,
        max_length=4,
        description="1 to 4 search queries, each for one part of the question.",
    )


class Grade(BaseModel):
    """The grader's view of the retrieved chunks."""

    relevant: list[str] = Field(description="Ids of the chunks that help answer the query.")
    sufficient: bool = Field(description="True if the relevant chunks answer the query fully.")
    missing: str = Field(default="", description="What is still missing, if not sufficient.")


class Rewrite(BaseModel):
    """A new search query."""

    query: str = Field(description="A new search query, at most 20 words.")


class Finding(BaseModel):
    """What the research loop found for one query.

    Attributes:
        query: The planned query.
        queries: Every query tried (the planned one first).
        chunk_ids: Relevant chunks, best first.
        tiers: Tiers searched in the last round.
        sufficient: Whether the grader judged the evidence sufficient.
        loops: Retrieve-grade rounds used.
    """

    query: str
    queries: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    tiers: list[int] = Field(default_factory=list)
    sufficient: bool = False
    loops: int = 0


class Conflict(BaseModel):
    """Two sources stating different values for the same thing."""

    topic: str = Field(description="What the values are about, e.g. 'F1 on RT-BENE'.")
    dissertation_value: str = Field(description="The value as written in the dissertation.")
    dissertation_chunk: str = Field(description="Id of the dissertation chunk stating it.")
    other_value: str = Field(description="The value as written in the other source.")
    other_chunk: str = Field(description="Id of the other chunk stating it.")


class Conflicts(BaseModel):
    """Conflicts found in the evidence (usually none)."""

    items: list[Conflict] = Field(default_factory=list)


class Sentence(BaseModel):
    """One sentence of an answer and the chunks it is based on."""

    text: str = Field(description="One sentence, at most 35 words.")
    chunk_ids: list[str] = Field(min_length=1, description="Ids of the chunks it uses.")


class AnswerDraft(BaseModel):
    """The writer's answer, sentence by sentence."""

    sentences: list[Sentence] = Field(min_length=1, max_length=8)


class Citation(BaseModel):
    """A numbered source reference of an answer.

    Attributes:
        n: Number used in the answer text ("[1]").
        chunk_id: The cited chunk.
        source_id: Its source.
        label: How it is shown, e.g. ``Dissertation §4.2, p. 57``.
        tier: Source tier.
        text: The chunk text (shown on hover in the dashboard).
    """

    n: int
    chunk_id: str
    source_id: str
    label: str
    tier: int
    text: str = ""


class Answer(BaseModel):
    """The final answer of a turn.

    Attributes:
        question: The question as asked.
        text: The answer with ``[n]`` citation markers.
        sentences: The verified sentences.
        citations: Numbered references.
        conflicts: Disagreements between sources that the answer resolves.
        abstained: True if the library does not answer the question.
        reason: Why it abstained.
        dropped: Sentences removed by the fact-check.
        agent: ``graph`` or ``prebuilt``.
    """

    question: str
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    abstained: bool = False
    reason: str = ""
    dropped: int = 0
    agent: str = "graph"


class Turn(BaseModel):
    """One exchange of a conversation (the graph's memory)."""

    question: str
    answer: str
