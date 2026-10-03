"""Data model of ask: what the models return and what the answer carries."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Understanding(BaseModel):
    """How the model reads a question.

    Attributes:
        on_topic: Whether the library could answer it at all.
        options: Readings to choose from if the question is ambiguous (else empty).
        search: The question as a search query.
    """

    on_topic: bool = Field(
        description="False for anything this library cannot answer: general knowledge, "
        "other topics, personal questions, requests to do something."
    )
    options: list[str] = Field(
        default_factory=list,
        description="Only if the question could clearly mean different things in this "
        "library: 2 to 4 short readings to choose from. Otherwise empty.",
    )
    search: str = Field(description="The question as a search query of at most 20 words.")


class Grade(BaseModel):
    """The grader's view of the retrieved chunks."""

    relevant: list[str] = Field(description="Ids of the chunks that help answer the query.")
    sufficient: bool = Field(description="True if the relevant chunks answer the query fully.")
    missing: str = Field(default="", description="What is still missing, if not sufficient.")


class Rewrite(BaseModel):
    """A new search query."""

    query: str = Field(description="A new search query, at most 20 words.")


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
        text: The chunk text.
    """

    n: int
    chunk_id: str
    source_id: str
    label: str
    text: str = ""


class Answer(BaseModel):
    """The final answer to a question.

    Attributes:
        question: The question as asked.
        text: The answer with ``[n]`` citation markers.
        sentences: The cited sentences.
        citations: Numbered references.
        abstained: True if the library does not answer the question.
        reason: Why it abstained.
        dropped: Sentences removed because the evidence did not support them.
        agent: ``graph`` or ``agent``.
    """

    question: str
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    reason: str = ""
    dropped: int = 0
    agent: str = "graph"


class SentenceVerdict(BaseModel):
    """The judge's verdict on one answer sentence."""

    sentence: int = Field(description="The sentence number, as shown.")
    supported: bool = Field(
        description="True only if the cited evidence states what the sentence says."
    )
    reason: str = Field(default="", description="One short phrase.")


class SentenceVerdicts(BaseModel):
    """The judge's verdicts on all sentences of an answer."""

    verdicts: list[SentenceVerdict]
