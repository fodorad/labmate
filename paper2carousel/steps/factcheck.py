"""Step 6 (EVALUATOR-OPTIMIZER): check every bullet against its evidence, rewrite failures.

Round structure (each phase uses one model, so each round costs at most two swaps)::

    deterministic checks ─▶ judge all slides (critic) ─▶ rewrite failing slides (writer) ─┐
            ▲                                                                              │
            └──────────────────────── at most ``max_rounds`` times ◀───────────────────────┘

Bullets still failing after the budget are dropped, and a slide without bullets is
dropped. Everything is recorded in :class:`FactCheckReport`, which feeds the headline
metric: the share of unsupported bullets before vs after the loop.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from paper2carousel.llm.structured import structured_chat
from paper2carousel.parallel import parallel_map
from paper2carousel.phases import ModelSwitcher
from paper2carousel.schemas import (
    BulletCheck,
    ClaimCard,
    Claims,
    FactChecked,
    FactCheckReport,
    SlideText,
    SlideVerdicts,
    WrittenSlides,
)
from paper2carousel.steps.extract import normalize
from paper2carousel.steps.llm import LLM, load_prompt
from paper2carousel.steps.write import check_slide, format_cards

_NUMBER = re.compile(r"(?<![a-z\d.,])\d+(?:[.,]\d+)*")
"""A number not glued to a preceding letter: "0.744" counts, the 1 in "F1" does not."""


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


def format_for_judge(slide: SlideText, by_id: dict[str, ClaimCard]) -> str:
    """Numbered bullets, each followed by the evidence quotes it cites.

    Args:
        slide: Slide to judge.
        by_id: Claim cards by id.

    Returns:
        Text block for the judge prompt.
    """
    blocks = []
    for i, bullet in enumerate(slide.bullets, start=1):
        quotes = "\n".join(
            f'   evidence ({cid}): "{by_id[cid].evidence_quote}"'
            for cid in bullet.claim_ids
            if cid in by_id
        )
        blocks.append(f"Bullet {i}: {bullet.text}\n{quotes}")
    return "\n\n".join(blocks)


def judge_slide(slide: SlideText, by_id: dict[str, ClaimCard], judge: LLM) -> SlideVerdicts:
    """One critic call per slide, exactly one verdict per bullet.

    Args:
        slide: Slide to judge.
        by_id: Claim cards by id.
        judge: Critic model settings.

    Returns:
        Verdicts, one per bullet.
    """
    n = len(slide.bullets)

    def check(v: SlideVerdicts) -> list[str]:
        numbers = sorted(x.bullet for x in v.verdicts)
        expected = list(range(1, n + 1))
        return [] if numbers == expected else [f"give exactly one verdict for bullets {expected}"]

    prompt = load_prompt("judge").format(bullets=format_for_judge(slide, by_id))
    return structured_chat(judge.backend, judge.request(prompt), SlideVerdicts, check=check)


def rewrite_slide(
    slide: SlideText, failing: list[BulletCheck], cards: list[ClaimCard], writer: LLM
) -> SlideText:
    """Ask the writer to fix a slide, given the judge's and the checks' findings.

    Args:
        slide: Current slide.
        failing: Failing bullet checks of this slide.
        cards: The slide's claim cards (the only facts it may use).
        writer: Writer model settings.

    Returns:
        The rewritten slide (validated with the writer's rules).
    """
    current = f"Title: {slide.title}\n" + "\n".join(
        f"Bullet {i}: {b.text} (claims: {', '.join(b.claim_ids)})"
        for i, b in enumerate(slide.bullets, start=1)
    )
    problems = "\n".join(
        f"Bullet {c.bullet}: " + "; ".join([*c.problems, *([c.reason] if c.reason else [])])
        for c in failing
    )
    prompt = load_prompt("rewrite").format(
        slide=current, problems=problems, claims=format_cards(cards)
    )
    allowed = {c.id for c in cards}
    fixed = structured_chat(
        writer.backend,
        writer.request(prompt),
        SlideText,
        check=lambda s: check_slide(s, allowed),
    )
    return fixed.model_copy(update={"title": slide.title})  # titles aren't fact-checked


class FactCheckLoop(BaseModel):
    """State of the evaluator-optimizer loop between rounds.

    Both engines drive the loop through :func:`judge_pending`, :func:`rewrite_pending`
    and :func:`should_rewrite`: the plain engine in a ``while`` loop, the LangGraph
    engine as a cycle of two nodes.

    Attributes:
        hook: Carousel hook (passed through).
        slides: Current slide texts.
        latest: Latest checks per slide index.
        pending: Slide indices that still fail (to rewrite, then re-judge).
        report: Audit trail so far.
        exempt: Names bullets may use without quoting them (from the paper title).
    """

    hook: str
    slides: list[SlideText]
    latest: dict[int, list[BulletCheck]] = Field(default_factory=dict)
    pending: list[int]
    report: FactCheckReport = Field(default_factory=lambda: FactCheckReport(rounds=[]))
    exempt: list[str] = Field(default_factory=list)


def start_loop(written: WrittenSlides, paper_title: str = "") -> FactCheckLoop:
    """Initial loop state: every slide is pending judgement.

    Args:
        written: Slides from the writer.
        paper_title: Its names (e.g. the method's) need no quote.

    Returns:
        The loop state.
    """
    return FactCheckLoop(
        hook=written.hook,
        slides=list(written.slides),
        pending=list(range(len(written.slides))),
        exempt=sorted(title_names(paper_title)),
    )


def judge_pending(
    loop: FactCheckLoop, by_id: dict[str, ClaimCard], judge: LLM, workers: int = 2
) -> FactCheckLoop:
    """Judge the pending slides and record a round.

    Args:
        loop: Current state.
        by_id: Claim cards by id.
        judge: Critic model settings.
        workers: Concurrent calls.

    Returns:
        The new state; ``pending`` holds the slides that still fail.
    """
    slides = loop.slides
    verdicts = parallel_map(lambda i: judge_slide(slides[i], by_id, judge), loop.pending, workers)
    latest = dict(loop.latest)
    for i, slide_verdicts in zip(loop.pending, verdicts, strict=True):
        by_bullet = {v.bullet: v for v in slide_verdicts.verdicts}
        latest[i] = [
            BulletCheck(
                slide=i + 1,
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
            for b, bullet in enumerate(slides[i].bullets, start=1)
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
    """True if some slide still fails and the rewrite budget isn't used up.

    Args:
        loop: State after a judging round.
        max_rounds: Rewrite rounds allowed after the first check.

    Returns:
        Whether to rewrite and judge again.
    """
    return bool(loop.pending) and len(loop.report.rounds) <= max_rounds


def rewrite_pending(
    loop: FactCheckLoop,
    slide_claims: list[list[str]],
    by_id: dict[str, ClaimCard],
    writer: LLM,
    workers: int = 2,
) -> FactCheckLoop:
    """Rewrite the failing slides with the judge's findings.

    Args:
        loop: State after a judging round.
        slide_claims: Claim ids assigned to each slide by the outline (rewrite scope).
        by_id: Claim cards by id.
        writer: Writer model settings.
        workers: Concurrent calls.

    Returns:
        The new state with rewritten slides (still pending until judged).
    """
    rewritten = parallel_map(
        lambda i: rewrite_slide(
            loop.slides[i],
            [c for c in loop.latest[i] if not c.passed],
            [by_id[c] for c in slide_claims[i]],
            writer,
        ),
        loop.pending,
        workers,
    )
    slides = list(loop.slides)
    for i, slide in zip(loop.pending, rewritten, strict=True):
        slides[i] = slide
    return loop.model_copy(update={"slides": slides})


def finish_loop(loop: FactCheckLoop) -> FactChecked:
    """Drop bullets that still fail, and slides left without bullets.

    Args:
        loop: Final state.

    Returns:
        Corrected slides and the full report.
    """
    report = loop.report.model_copy(deep=True)
    final: list[SlideText] = []
    for i, slide in enumerate(loop.slides):
        keep = [b for b, c in zip(slide.bullets, loop.latest[i], strict=True) if c.passed]
        report.dropped.extend(c for c in loop.latest[i] if not c.passed)
        if keep:
            final.append(slide.model_copy(update={"bullets": keep}))
        else:
            report.dropped_slides.append(i + 1)
    return FactChecked(slides=WrittenSlides(hook=loop.hook, slides=final), report=report)


def fact_check(
    written: WrittenSlides,
    slide_claims: list[list[str]],
    claims: Claims,
    writer: LLM,
    judge: LLM,
    switcher: ModelSwitcher,
    max_rounds: int = 2,
    workers: int = 2,
    paper_title: str = "",
) -> FactChecked:
    """Run the evaluator-optimizer loop over all slides.

    Args:
        written: Slides from the writer.
        slide_claims: Claim ids assigned to each slide by the outline (rewrite scope).
        claims: All claim cards.
        writer: Writer model settings (rewrites).
        judge: Critic model settings (verdicts); should be a different model.
        switcher: Keeps one model in memory per phase.
        max_rounds: Rewrite rounds after the first check.
        workers: Concurrent calls within a phase.
        paper_title: Its names need no quote.

    Returns:
        Corrected slides and the per-round report.
    """
    by_id = {c.id: c for c in claims.cards}
    loop = start_loop(written, paper_title)
    while True:
        switcher.use(judge.model)
        loop = judge_pending(loop, by_id, judge, workers)
        if not should_rewrite(loop, max_rounds):
            return finish_loop(loop)
        switcher.use(writer.model)
        loop = rewrite_pending(loop, slide_claims, by_id, writer, workers)
