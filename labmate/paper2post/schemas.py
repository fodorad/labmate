"""paper2post data model: the post draft and the finished post."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from labmate.core.schemas import Bullet, FactCheckReport

IconName = Literal[
    "alert-triangle", "arrows-shuffle", "bolt", "book", "brain", "bulb", "chart-bar",
    "clock", "cpu", "database", "eye", "file-text", "flask", "git-merge", "lock",
    "math-function", "message-question", "microphone", "photo", "puzzle", "rocket", "route",
    "scale", "search", "stack-2", "target", "trophy", "users", "video", "world",
]  # fmt: skip
"""The bundled icons (Tabler Icons, MIT; ``paper2post/icons``)."""


class PostDraft(BaseModel):
    """A LinkedIn post drafted from the fact-checked cards."""

    hook: str = Field(description="First line, at most 15 words, makes people open the post.")
    takeaways: list[Bullet] = Field(
        description="3 to 5 sentences explaining the paper, each citing its claim ids.",
        min_length=3,
        max_length=5,
    )
    question: str = Field(description="A closing question to readers, with no factual claims.")


class Link(BaseModel):
    """A labelled link under the post."""

    label: str
    url: str


class Post(BaseModel):
    """The fact-checked post, its icons, its links and its audit trail.

    Attributes:
        hook: First line.
        takeaways: The sentences that passed the fact-check.
        question: Closing question.
        icons: One icon per takeaway .
        links: The paper, its code if it names a repository, and your own links.
        report: Fact-check audit of the takeaways.
    """

    hook: str
    takeaways: list[Bullet]
    question: str
    icons: list[IconName] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    report: FactCheckReport
