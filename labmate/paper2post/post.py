"""The post's content: text drafted from the fact-checked cards, icons, links.

The post body (3 to 5 sentences telling the paper's story) goes through the same
evaluator-optimizer loop as the overview's cards, so the post can't state anything the
overview couldn't; a sentence that stays unsupported is dropped. The icons are picked
afterwards, one per surviving sentence, from a fixed set.
"""

from __future__ import annotations

import re

from langchain_core.runnables import RunnableConfig

from labmate.core.factcheck import INLINE_ID, fact_check
from labmate.core.lc import RecordedChatModel, prompt, structured
from labmate.core.schemas import Card, Cards, Claims, Paper
from labmate.paper2post.prompts import load_prompt
from labmate.paper2post.schemas import IconChoice, IconName, Link, Post, PostDraft

GROUP = 3
"""Sentences judged per fact-check call (the loop checks short cards)."""

_REPO = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+")
"""A GitHub repository URL in the paper's text."""


def _format_cards(cards: Cards) -> str:
    return "\n".join(
        f"- {b.text} [{', '.join(b.claim_ids)}]" for c in cards.cards for b in c.bullets
    )


def write_post(
    cards: Cards,
    claims: Claims,
    paper: Paper,
    writer: RecordedChatModel,
    judge: RecordedChatModel,
    max_rounds: int = 1,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> Post:
    """Draft the post from the overview's cards, then fact-check its sentences.

    Args:
        cards: The fact-checked cards.
        claims: Claim cards.
        paper: The paper.
        writer: The writer model.
        judge: The critic model.
        max_rounds: Rewrite rounds for the takeaways.
        workers: Concurrent calls.
        config: The calling step's config (callbacks).

    Returns:
        The post with only supported takeaways, and the audit trail.
    """
    used = sorted({cid for c in cards.cards for b in c.bullets for cid in b.claim_ids})
    known = set(used)

    def check(d: PostDraft) -> list[str]:
        problems = [
            f"takeaway {i} cites unknown claims {sorted(set(t.claim_ids) - known)}"
            for i, t in enumerate(d.takeaways, start=1)
            if set(t.claim_ids) - known
        ]
        return problems + [
            f"takeaway {i} has claim ids in its text; list them in claim_ids only"
            for i, t in enumerate(d.takeaways, start=1)
            if INLINE_ID.search(t.text)
        ]

    draft = (prompt(load_prompt("post")) | structured(writer, PostDraft, check=check)).invoke(
        {"title": paper.title, "cards": _format_cards(cards)}, config
    )
    groups = [draft.takeaways[i : i + GROUP] for i in range(0, len(draft.takeaways), GROUP)]
    checked = fact_check(
        Cards(cards=[Card(title=draft.hook, bullets=g) for g in groups]),
        [used] * len(groups),
        claims,
        writer,
        judge,
        max_rounds,
        workers,
        paper.title,
        config,
    )
    return Post(
        hook=draft.hook,
        takeaways=[b for c in checked.cards.cards for b in c.bullets],
        question=draft.question,
        report=checked.report,
    )


def pick_icons(
    post: Post, model: RecordedChatModel, config: RunnableConfig | None = None
) -> list[IconName]:
    """One icon per takeaway, from the bundled set.

    Args:
        post: The fact-checked post.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        Icon names, one per takeaway.
    """
    n = len(post.takeaways)

    def check(choice: IconChoice) -> list[str]:
        return [] if len(choice.icons) == n else [f"give exactly {n} icons, one per sentence"]

    sentences = "\n".join(f"{i}. {t.text}" for i, t in enumerate(post.takeaways, start=1))
    chooser = structured(model, IconChoice, check=check)
    return (prompt(load_prompt("icons")) | chooser).invoke({"sentences": sentences}, config).icons


def post_links(paper: Paper, own: dict[str, str]) -> list[Link]:
    """The links under the post: the paper, its code (if it names a repository), yours.

    Args:
        paper: The paper.
        own: Your links (label -> URL), from ``[post].links``.

    Returns:
        Links in that order.
    """
    links = [Link(label="Paper", url=paper.url)] if paper.url else []
    text = "\n".join(s.text for s in paper.sections)
    if repo := _REPO.search(text):
        links.append(Link(label="Code", url=repo.group(0).rstrip(".")))
    return links + [Link(label=label, url=url) for label, url in own.items()]
