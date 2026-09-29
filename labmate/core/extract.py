"""Claim extraction (PARALLELISATION, sectioning): claim cards section by section.

Each section is an independent unit, so units run concurrently. Every claim must carry a
verbatim evidence quote, and a deterministic guard drops claims whose quote cannot be
found in the section text: the cheapest possible hallucination filter, applied before
any LLM judge.
"""

from __future__ import annotations

import re
import unicodedata

from langchain_core.runnables import RunnableConfig
from rapidfuzz import fuzz

from labmate.core.lc import RecordedChatModel, batch_map, prompt, structured
from labmate.core.prompts import load_prompt
from labmate.core.schemas import ClaimCard, ClaimDraft, Claims, Paper, Section, SectionClaims

CHUNK_CHARS = 6000
"""Longer sections are split at line breaks into chunks of at most this size."""

MIN_SECTION_CHARS = 200
"""Sections shorter than this (e.g. a lone heading) are skipped."""

MAX_CLAIMS = 4
"""Claims requested per chunk."""

QUOTE_MATCH = 90.0
"""Minimum fuzzy-match score (0-100) of a quote against its section."""


def normalize(text: str) -> str:
    """Normalise PDF text for matching: unfold ligatures, join broken words, squash spaces.

    Args:
        text: Raw text.

    Returns:
        Lower-cased, whitespace-normalised text.
    """
    text = unicodedata.normalize("NFKC", text)  # ligatures: "ﬁ" -> "fi"
    text = re.sub(r"-\s*\n\s*", "", text)
    return " ".join(text.split()).lower()


_NUMERIC = re.compile(r"\S*\d\S*")
_ELLIPSIS = re.compile(r"\[?(?:\.\s*){3}\]?|…")


def numeric_tokens(text: str) -> list[str]:
    """Tokens containing a digit (``84.6%``, ``P100``, ``N=6``), stripped of punctuation.

    Args:
        text: Normalised text.

    Returns:
        The tokens, in order.
    """
    return [tok.strip(".,;:()[]") for tok in _NUMERIC.findall(text)]


def quote_score(quote: str, text: str) -> float:
    """How well ``quote`` occurs somewhere in ``text`` (0-100).

    Fuzzy matching tolerates PDF extraction noise in the wording, but facts must not
    drift: every token containing a digit has to appear exactly in the text, otherwise
    the score is 0. Without this, "eight V100 GPUs" matches "eight P100 GPUs" at ~97.
    Quotes shortened with an ellipsis are checked piece by piece (each piece of at least
    three words must match; the worst piece decides).

    Args:
        quote: Claimed verbatim quote.
        text: Section text.

    Returns:
        Fuzzy partial-match score; 100 means an exact (normalised) substring.
    """
    t = normalize(text)
    fragments = [f for f in _ELLIPSIS.split(normalize(quote)) if len(f.split()) >= 3]
    if not fragments:
        return 0.0
    return min(_fragment_score(f.strip(), t) for f in fragments)


def _fragment_score(q: str, t: str) -> float:
    if q in t:
        return 100.0
    if any(tok not in t for tok in numeric_tokens(q)):
        return 0.0
    return float(fuzz.partial_ratio(q, t))


def chunk_section(section: Section, max_chars: int = CHUNK_CHARS) -> list[Section]:
    """Split a long section at line breaks; short sections are returned as is.

    Args:
        section: A paper section.
        max_chars: Maximum chunk size.

    Returns:
        One or more sections with the same title and page.
    """
    if len(section.text) <= max_chars:
        return [section]
    chunks, current = [], ""
    for line in section.text.splitlines(keepends=True):
        if current and len(current) + len(line) > max_chars:
            chunks.append(current)
            current = ""
        current += line
    chunks.append(current)
    return [Section(title=section.title, page=section.page, text=c) for c in chunks]


def extraction_units(paper: Paper) -> list[Section]:
    """Sections long enough to hold claims, split into prompt-sized chunks.

    Args:
        paper: Ingested paper.

    Returns:
        Units in reading order; one extraction call each.
    """
    return [
        chunk
        for section in paper.sections
        if len(section.text) >= MIN_SECTION_CHARS
        for chunk in chunk_section(section)
    ]


def extract_unit(
    title: str, unit: Section, model: RecordedChatModel, config: RunnableConfig | None = None
) -> list[ClaimDraft]:
    """Draft claims for one unit (one model call).

    Args:
        title: Paper title.
        unit: Section chunk.
        model: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        Claim drafts, not yet verified.
    """
    chain = prompt(load_prompt("extract")) | structured(model, SectionClaims)
    values = {
        "max_claims": MAX_CLAIMS,
        "title": title,
        "section": unit.title,
        "page": unit.page,
        "text": unit.text,
    }
    return chain.invoke(values, config).claims


def assemble_claims(units: list[Section], drafts: list[list[ClaimDraft]]) -> Claims:
    """Verify every draft's quote against its unit and number the survivors.

    Args:
        units: Extraction units, in reading order.
        drafts: Claim drafts per unit, same order.

    Returns:
        Verified claim cards (ids ``c01``... in reading order) and the rejected drafts.
    """
    cards: list[ClaimCard] = []
    rejected: list[ClaimDraft] = []
    for unit, unit_drafts in zip(units, drafts, strict=True):
        for draft in unit_drafts:
            score = quote_score(draft.evidence_quote, unit.text)
            if score < QUOTE_MATCH:
                rejected.append(draft)
                continue
            cards.append(
                ClaimCard(
                    id=f"c{len(cards) + 1:02d}",
                    claim=draft.claim,
                    evidence_quote=draft.evidence_quote,
                    kind=draft.kind,
                    section=unit.title,
                    page=unit.page,
                    match=round(score, 1),
                )
            )
    return Claims(cards=cards, rejected=rejected)


def extract_claims(
    paper: Paper, model: RecordedChatModel, workers: int = 2, config: RunnableConfig | None = None
) -> Claims:
    """Extract and verify claim cards for the whole paper (parallel over units).

    Args:
        paper: Ingested paper.
        model: The writer model.
        workers: Concurrent extraction calls.
        config: The calling step's config (callbacks).

    Returns:
        Verified claim cards (ids ``c01``... in reading order) and the rejected drafts.
    """
    units = extraction_units(paper)
    drafts = batch_map(
        lambda u, cfg: extract_unit(paper.title, u, model, cfg), units, config, workers
    )
    return assemble_claims(units, drafts)
