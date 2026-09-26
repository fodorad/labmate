"""Step 2 (PARALLELISATION, sectioning): extract claim cards section by section.

Each section is an independent unit, so units run concurrently. Every claim must carry a
verbatim evidence quote, and a deterministic guard drops claims whose quote cannot be
found in the section text: the cheapest possible hallucination filter, applied before
any LLM judge.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from paper2carousel.llm.structured import structured_chat
from paper2carousel.parallel import parallel_map
from paper2carousel.schemas import ClaimCard, ClaimDraft, Claims, Paper, Section, SectionClaims
from paper2carousel.steps.llm import LLM, load_prompt

CHUNK_CHARS = 6000
"""Longer sections are split at line breaks into chunks of at most this size."""

MIN_SECTION_CHARS = 200
"""Sections shorter than this (e.g. a lone heading) are skipped."""

MAX_CLAIMS = 4
"""Claims requested per chunk."""

QUOTE_MATCH = 90.0
"""Minimum fuzzy-match score (0-100) of a quote against its section."""


def normalize(text: str) -> str:
    """Normalise PDF text for matching: join hyphenated line breaks, collapse whitespace.

    Args:
        text: Raw text.

    Returns:
        Lower-cased, whitespace-normalised text.
    """
    text = re.sub(r"-\s*\n\s*", "", text)
    return " ".join(text.split()).lower()


_NUMERIC = re.compile(r"\S*\d\S*")


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

    Args:
        quote: Claimed verbatim quote.
        text: Section text.

    Returns:
        Fuzzy partial-match score; 100 means an exact (normalised) substring.
    """
    q, t = normalize(quote), normalize(text)
    if len(q.split()) < 3:
        return 0.0
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


def extract_unit(title: str, unit: Section, llm: LLM) -> list[ClaimDraft]:
    """Draft claims for one unit (one model call).

    Args:
        title: Paper title.
        unit: Section chunk.
        llm: Model settings.

    Returns:
        Claim drafts, not yet verified.
    """
    prompt = load_prompt("extract").format(
        max_claims=MAX_CLAIMS,
        title=title,
        section=unit.title,
        page=unit.page,
        text=unit.text,
    )
    return structured_chat(llm.backend, llm.request(prompt), SectionClaims).claims


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


def extract_claims(paper: Paper, llm: LLM, workers: int = 2) -> Claims:
    """Extract and verify claim cards for the whole paper (parallel over units).

    Args:
        paper: Ingested paper.
        llm: Model settings.
        workers: Concurrent extraction calls.

    Returns:
        Verified claim cards (ids ``c01``... in reading order) and the rejected drafts.
    """
    units = extraction_units(paper)
    drafts = parallel_map(lambda u: extract_unit(paper.title, u, llm), units, workers)
    return assemble_claims(units, drafts)
