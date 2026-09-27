"""Where and when the paper was published: venue and date, read from its first page.

The model reads the venue (journal or conference) and the publication date off the first
page and the arXiv notes; code accepts them only if they appear verbatim in that text, so
a venue can't be guessed from the topic. The year falls back to the arXiv submission date
(or the PDF's creation date) when the paper prints none.
"""

from __future__ import annotations

import re

from labmate.core.llm.structured import StructuredOutputError, structured_chat
from labmate.core.model import LLM
from labmate.paper2flow.prompts import load_prompt
from labmate.paper2flow.schemas import Paper, PublicationDraft

_YEAR = re.compile(r"\b(19[5-9]\d|20\d\d)\b")


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def check_publication(draft: PublicationDraft, source: str) -> list[str]:
    """Rules for the model's answer: both fields verbatim from the source, a year in the date.

    Args:
        draft: Venue and date as read by the model.
        source: The first-page text and arXiv notes it was given.

    Returns:
        Problems; empty if the answer is acceptable.
    """
    problems = []
    text = _normal(source)
    for name, value in (("venue", draft.venue), ("date", draft.date)):
        if value.strip() and _normal(value) not in text:
            problems.append(f"the {name} {value!r} is not written in the source; copy it exactly")
    if draft.date.strip() and not _YEAR.search(draft.date):
        problems.append(f"the date {draft.date!r} has no year")
    if re.search(r"\barxiv\b", draft.venue, re.IGNORECASE):
        problems.append("arXiv is a preprint server, not a venue; use '' if there is no venue")
    return problems


def read_publication(paper: Paper, first_page: str, llm: LLM) -> PublicationDraft:
    """Ask the model for the venue and publication date, checked against the source text.

    Args:
        paper: The paper (title and arXiv notes).
        first_page: Text of the PDF's first page.
        llm: Model settings.

    Returns:
        The checked answer; empty fields if the model never gave an acceptable one.
    """
    source = "\n".join(t for t in (paper.notes, first_page) if t)
    if not source.strip():
        return PublicationDraft(venue="", date="")
    prompt = load_prompt("publication").format(title=paper.title, source=source)
    try:
        return structured_chat(
            llm.backend,
            llm.request(prompt),
            PublicationDraft,
            check=lambda d: check_publication(d, source),
        )
    except StructuredOutputError:
        return PublicationDraft(venue="", date="")


def with_publication(paper: Paper, draft: PublicationDraft) -> Paper:
    """The paper with venue, date and year filled in.

    The year comes from the printed date, else from the venue (``NIPS 2017``), else from
    what ingestion found (arXiv submission or PDF creation date).

    Args:
        paper: Ingested paper.
        draft: Checked venue and date.

    Returns:
        A copy with ``venue``, ``date`` and ``year`` set.
    """
    year = next(
        (int(m.group(0)) for text in (draft.date, draft.venue) if (m := _YEAR.search(text))),
        paper.year,
    )
    return paper.model_copy(
        update={
            "venue": " ".join(draft.venue.split()),
            "date": " ".join(draft.date.split()) or paper.date,
            "year": year,
        }
    )
