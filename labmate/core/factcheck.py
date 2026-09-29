"""Fact-check loop (EVALUATOR-OPTIMIZER): check bullets against evidence, rewrite failures.

Round structure (each phase uses one model, so each round costs at most two swaps)::

    deterministic checks ─▶ judge all cards (critic) ─▶ rewrite failing cards (writer) ─┐
            ▲                                                                            │
            └─────────────────────── at most ``max_rounds`` times ◀──────────────────────┘

Bullets still failing after the budget are dropped, and a card without bullets is
dropped. Everything is recorded in :class:`FactCheckReport`, which feeds the headline
metric: the share of unsupported bullets before vs after the loop.

The loop is driven by four functions (:func:`start_loop`, :func:`judge_pending`,
:func:`should_rewrite` / :func:`rewrite_pending`, :func:`finish_loop`). paper2flow and
paper2post run them as a bounded loop inside one chain step (:func:`fact_check`); the ask
agent wires the same functions as a LangGraph subgraph.
"""

from __future__ import annotations

import re

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from labmate.core.extract import normalize
from labmate.core.lc import RecordedChatModel, batch_map, prompt, structured
from labmate.core.prompts import load_prompt
from labmate.core.schemas import (
    BulletCheck,
    Card,
    Cards,
    CardVerdicts,
    ClaimCard,
    Claims,
    FactChecked,
    FactCheckReport,
)

_NUMBER = re.compile(r"(?<![a-z\d.,])\d+(?:[.,]\d+)*")
"""A number not glued to a preceding letter: "0.744" counts, the 1 in "F1" does not."""


INLINE_ID = re.compile(r"\[?\bc\d{2,}\b\]?")
"""A claim id written into the text ("[c03]"), which belongs in ``claim_ids``."""

MAX_WORDS = 30
"""Hard limit per bullet (the prompt asks for 25; a little slack avoids needless retries)."""


def check_card(card: Card, allowed_ids: set[str]) -> list[str]:
    """Rules for a written card.

    Args:
        card: Candidate card.
        allowed_ids: Claim ids assigned to this card by the outline.

    Returns:
        Problems; empty if valid.
    """
    problems = []
    if len(card.title.split()) > 10:
        problems.append("the title has more than 10 words")
    for i, bullet in enumerate(card.bullets, start=1):
        if len(bullet.text.split()) > MAX_WORDS:
            problems.append(f"bullet {i} has more than {MAX_WORDS} words")
        if INLINE_ID.search(bullet.text):
            problems.append(f"bullet {i} has claim ids in its text; list them in claim_ids only")
        foreign = sorted(set(bullet.claim_ids) - allowed_ids)
        if foreign:
            problems.append(f"bullet {i} cites claims not on this card: {foreign}")
    return problems


def format_evidence(cards: list[ClaimCard]) -> str:
    """Claim cards with their evidence quotes, as shown to the writer.

    Args:
        cards: The claim cards a card may use.

    Returns:
        Text block.
    """
    return "\n".join(
        f'{c.id} [{c.kind}] {c.claim}\n    evidence: "{c.evidence_quote}"' for c in cards
    )


def numbers_in(text: str) -> set[str]:
    """Numbers appearing in ``text`` (``28.4``, ``3.5``, ``8``), thousands separators removed.

    Digits that are part of a name (``F1``, ``ResNet50``, ``GPT-4o``'s letters aside) are
    not numbers here; names are the judge's job, amounts are this check's.

    Args:
        text: Any text.

    Returns:
        The set of number strings.
    """
    return {n.replace(",", "") for n in _NUMBER.findall(normalize(text))}


GENERIC_NAMES = frozenset(
    "f1 bleu rouge auc roc map mae mse rmse cnn cnns rnn rnns lstm gru gpu gpus tpu tpus cpu "
    "rgb ai ml llm llms nlp cv sota api fps".split()
)
"""Acronyms common enough to use without a quote naming them."""

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*[A-Za-z0-9]")


def names_in(text: str) -> set[str]:
    """Identifiers in ``text``, lower-cased: ``DenseNet121``, ``MRL``, ``TalkingFace``.

    Model, dataset and method names are what a paraphrase most easily gets wrong.

    Args:
        text: Any text.

    Returns:
        Tokens that mix letters and digits, are all capitals, or are CamelCase.
    """
    names = set()
    for token in _TOKEN.findall(text):
        letters = [c for c in token if c.isalpha()]
        if (
            (any(c.isdigit() for c in token) and letters)
            or (len(letters) >= 2 and token.isupper())
            or re.search(r"[a-z][A-Z]", token)
        ):
            names.add(token.lower())
    return names


def deterministic_problems(
    text: str, cards: list[ClaimCard], exempt: frozenset[str] = frozenset()
) -> list[str]:
    """Checks that need no model: numbers and names in the bullet must be in its evidence.

    Evidence means the verified quotes, not the claim paraphrases, which are model output
    and can carry context the quote doesn't (a real run wrote "DenseNet121 achieves 0.9953
    on MRL" from a quote that names neither).

    Args:
        text: Bullet text.
        cards: The claim cards the bullet cites.
        exempt: Names that need no quote (e.g. the paper title's, like the method's name).

    Returns:
        Problems; empty if the bullet passes.
    """
    quotes = " ".join(c.evidence_quote for c in cards)
    problems = []
    missing = sorted(numbers_in(text) - numbers_in(quotes), key=lambda n: (len(n), n))
    if missing:
        problems.append(f"number(s) {', '.join(missing)} not in the cited evidence")
    evidence = normalize(quotes)
    unnamed = sorted(n for n in names_in(text) - GENERIC_NAMES - exempt if n not in evidence)
    if unnamed:
        problems.append(f"name(s) {', '.join(unnamed)} not in the cited evidence")
    return problems


def title_names(title: str) -> frozenset[str]:
    """Names from the paper title, which bullets may use without quoting them.

    Args:
        title: Paper title.

    Returns:
        Lower-cased names, including the parts of hyphenated ones.
    """
    names = names_in(title)
    return frozenset(names | {part for n in names for part in n.split("-") if part})


def format_for_judge(card: Card, by_id: dict[str, ClaimCard]) -> str:
    """Numbered bullets, each followed by the evidence quotes it cites.

    Args:
        card: Card to judge.
        by_id: Claim cards by id.

    Returns:
        Text block for the judge prompt.
    """
    blocks = []
    for i, bullet in enumerate(card.bullets, start=1):
        quotes = "\n".join(
            f'   evidence ({cid}): "{by_id[cid].evidence_quote}"'
            for cid in bullet.claim_ids
            if cid in by_id
        )
        blocks.append(f"Bullet {i}: {bullet.text}\n{quotes}")
    return "\n\n".join(blocks)


def judge_card(
    card: Card,
    by_id: dict[str, ClaimCard],
    judge: RecordedChatModel,
    config: RunnableConfig | None = None,
) -> CardVerdicts:
    """One critic call per card, exactly one verdict per bullet.

    Args:
        card: Card to judge.
        by_id: Claim cards by id.
        judge: The critic model.
        config: The calling step's config (callbacks).

    Returns:
        Verdicts, one per bullet.
    """
    expected = list(range(1, len(card.bullets) + 1))

    def check(v: CardVerdicts) -> list[str]:
        numbers = sorted(x.bullet for x in v.verdicts)
        return [] if numbers == expected else [f"give exactly one verdict for bullets {expected}"]

    chain = prompt(load_prompt("judge")) | structured(judge, CardVerdicts, check=check)
    return chain.invoke({"bullets": format_for_judge(card, by_id)}, config)


def rewrite_card(
    card: Card,
    failing: list[BulletCheck],
    evidence: list[ClaimCard],
    writer: RecordedChatModel,
    config: RunnableConfig | None = None,
) -> Card:
    """Ask the writer to fix a card, given the judge's and the checks' findings.

    Args:
        card: Current card.
        failing: Failing bullet checks of this card.
        evidence: The card's claim cards (the only facts it may use).
        writer: The writer model.
        config: The calling step's config (callbacks).

    Returns:
        The rewritten card (validated with the writer's rules), with its title kept.
    """
    current = f"Title: {card.title}\n" + "\n".join(
        f"Bullet {i}: {b.text} (claims: {', '.join(b.claim_ids)})"
        for i, b in enumerate(card.bullets, start=1)
    )
    problems = "\n".join(
        f"Bullet {c.bullet}: " + "; ".join([*c.problems, *([c.reason] if c.reason else [])])
        for c in failing
    )
    allowed = {c.id for c in evidence}
    chain = prompt(load_prompt("rewrite")) | structured(
        writer, Card, check=lambda c: check_card(c, allowed)
    )
    fixed = chain.invoke(
        {"card": current, "problems": problems, "claims": format_evidence(evidence)}, config
    )
    return fixed.model_copy(update={"title": card.title})  # titles aren't fact-checked


class FactCheckLoop(BaseModel):
    """State of the evaluator-optimizer loop between rounds.

    Attributes:
        cards: Current cards.
        latest: Latest checks per card index.
        pending: Card indices that still fail (to rewrite, then re-judge).
        report: Audit trail so far.
        exempt: Names bullets may use without quoting them (from the paper title).
    """

    cards: list[Card]
    latest: dict[int, list[BulletCheck]] = Field(default_factory=dict)
    pending: list[int]
    report: FactCheckReport = Field(default_factory=lambda: FactCheckReport(rounds=[]))
    exempt: list[str] = Field(default_factory=list)


def start_loop(written: Cards, paper_title: str = "") -> FactCheckLoop:
    """Initial loop state: every card is pending judgement.

    Args:
        written: Cards from the writer.
        paper_title: Its names (e.g. the method's) need no quote.

    Returns:
        The loop state.
    """
    return FactCheckLoop(
        cards=list(written.cards),
        pending=list(range(len(written.cards))),
        exempt=sorted(title_names(paper_title)),
    )


def judge_pending(
    loop: FactCheckLoop,
    by_id: dict[str, ClaimCard],
    judge: RecordedChatModel,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> FactCheckLoop:
    """Judge the pending cards (in parallel) and record a round.

    Args:
        loop: Current state.
        by_id: Claim cards by id.
        judge: The critic model.
        workers: Concurrent calls.
        config: The calling step's config (callbacks).

    Returns:
        The new state; ``pending`` holds the cards that still fail.
    """
    cards = loop.cards
    verdicts = batch_map(
        lambda i, cfg: judge_card(cards[i], by_id, judge, cfg), loop.pending, config, workers
    )
    latest = dict(loop.latest)
    for i, card_verdicts in zip(loop.pending, verdicts, strict=True):
        by_bullet = {v.bullet: v for v in card_verdicts.verdicts}
        latest[i] = [
            BulletCheck(
                card=i + 1,
                bullet=b,
                text=bullet.text,
                claim_ids=bullet.claim_ids,
                problems=deterministic_problems(
                    bullet.text,
                    [by_id[c] for c in bullet.claim_ids if c in by_id],
                    frozenset(loop.exempt),
                ),
                verdict=by_bullet[b].verdict,
                reason=by_bullet[b].reason,
            )
            for b, bullet in enumerate(cards[i].bullets, start=1)
        ]
    rounds = [*loop.report.rounds, [c for i in sorted(latest) for c in latest[i]]]
    return loop.model_copy(
        update={
            "latest": latest,
            "report": FactCheckReport(rounds=rounds),
            "pending": [i for i in sorted(latest) if any(not c.passed for c in latest[i])],
        }
    )


def should_rewrite(loop: FactCheckLoop, max_rounds: int) -> bool:
    """True if some card still fails and the rewrite budget isn't used up.

    Args:
        loop: State after a judging round.
        max_rounds: Rewrite rounds allowed after the first check.

    Returns:
        Whether to rewrite and judge again.
    """
    return bool(loop.pending) and len(loop.report.rounds) <= max_rounds


def rewrite_pending(
    loop: FactCheckLoop,
    card_claims: list[list[str]],
    by_id: dict[str, ClaimCard],
    writer: RecordedChatModel,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> FactCheckLoop:
    """Rewrite the failing cards (in parallel) with the judge's findings.

    Args:
        loop: State after a judging round.
        card_claims: Claim ids assigned to each card (rewrite scope).
        by_id: Claim cards by id.
        writer: The writer model.
        workers: Concurrent calls.
        config: The calling step's config (callbacks).

    Returns:
        The new state with rewritten cards (still pending until judged).
    """
    rewritten = batch_map(
        lambda i, cfg: rewrite_card(
            loop.cards[i],
            [c for c in loop.latest[i] if not c.passed],
            [by_id[c] for c in card_claims[i]],
            writer,
            cfg,
        ),
        loop.pending,
        config,
        workers,
    )
    cards = list(loop.cards)
    for i, card in zip(loop.pending, rewritten, strict=True):
        cards[i] = card
    return loop.model_copy(update={"cards": cards})


def finish_loop(loop: FactCheckLoop) -> FactChecked:
    """Drop bullets that still fail, and cards left without bullets.

    Args:
        loop: Final state.

    Returns:
        Corrected cards and the full report.
    """
    report = loop.report.model_copy(deep=True)
    final: list[Card] = []
    for i, card in enumerate(loop.cards):
        keep = [b for b, c in zip(card.bullets, loop.latest[i], strict=True) if c.passed]
        report.dropped.extend(c for c in loop.latest[i] if not c.passed)
        if keep:
            final.append(card.model_copy(update={"bullets": keep}))
        else:
            report.dropped_cards.append(i + 1)
    return FactChecked(cards=Cards(cards=final), report=report)


def fact_check(
    written: Cards,
    card_claims: list[list[str]],
    claims: Claims,
    writer: RecordedChatModel,
    judge: RecordedChatModel,
    max_rounds: int = 2,
    workers: int = 2,
    paper_title: str = "",
    config: RunnableConfig | None = None,
) -> FactChecked:
    """Run the evaluator-optimizer loop over all cards.

    Args:
        written: Cards from the writer.
        card_claims: Claim ids assigned to each card (rewrite scope).
        claims: All claim cards.
        writer: The writer model (rewrites).
        judge: The critic model (verdicts); should be a different model.
        max_rounds: Rewrite rounds after the first check.
        workers: Concurrent calls within a phase.
        paper_title: Its names need no quote.
        config: The calling step's config (callbacks).

    Returns:
        Corrected cards and the per-round report.
    """
    by_id = {c.id: c for c in claims.cards}
    loop = judge_pending(start_loop(written, paper_title), by_id, judge, workers, config)
    while should_rewrite(loop, max_rounds):
        loop = rewrite_pending(loop, card_claims, by_id, writer, workers, config)
        loop = judge_pending(loop, by_id, judge, workers, config)
    return finish_loop(loop)
