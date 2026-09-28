"""Evaluation of ask: retrieval quality without hand labels, answer quality on a golden set.

**Retrieval.** Every verified claim card is a (quote, page) pair that the quote guard
already found in the text. The writer model turns each claim into a question a reader
might ask; the chunks that contain the quote are the right answers. That gives a test set
of hundreds of questions without labelling anything, and recall@k / MRR for BM25, dense,
hybrid and hybrid with query rewriting.

**Answers** (see :func:`answer_report`): a golden set you write yourself, scored for
grounded answers, correct citations and correct refusals, for the LangGraph agent and the
prebuilt baseline.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from pydantic import BaseModel, Field

from labmate.ask.chunk import Chunk
from labmate.ask.evidence import evidence_cards
from labmate.ask.index import Index, Method
from labmate.ask.prompts import load_prompt
from labmate.ask.schemas import Answer
from labmate.ask.session import AskSession
from labmate.ask.verify import to_blocks
from labmate.core.extract import QUOTE_MATCH, quote_score
from labmate.core.factcheck import judge_pending, start_loop
from labmate.core.llm.structured import structured_chat
from labmate.core.schemas import ClaimCard

KS = (1, 3, 5, 10)
"""Cut-offs reported for recall."""

BATCH = 8
"""Claims turned into questions per model call."""

MIN_GOLD_SCORE = 60.0
"""A quote that straddles two chunks matches each only partly; below this it is skipped."""


class GeneratedQuestion(BaseModel):
    """A question for one claim card."""

    id: str = Field(description="The claim id, copied exactly.")
    question: str = Field(description="A question the claim answers, in your own words.")


class GeneratedQuestions(BaseModel):
    """Questions for a batch of claim cards."""

    items: list[GeneratedQuestion]


class RetrievalCase(BaseModel):
    """One test question.

    Attributes:
        claim_id: The claim it was made from.
        source_id: The claim's source.
        question: The question.
        gold: Ids of the chunks that contain the claim's quote.
    """

    claim_id: str
    source_id: str
    question: str
    gold: list[str]


class MethodScore(BaseModel):
    """Retrieval metrics of one method.

    Attributes:
        method: Method name.
        recall: Recall@k per cut-off (share of questions with a gold chunk in the top k).
        mrr: Mean reciprocal rank of the first gold chunk (top 10).
        n: Questions scored.
    """

    method: str
    recall: dict[int, float]
    mrr: float
    n: int


def _overlap(question: str, quote: str, n: int = 8) -> bool:
    """Whether the question copies ``n`` consecutive words of the quote."""
    q, t = question.lower().split(), quote.lower().split()
    grams = {" ".join(t[i : i + n]) for i in range(len(t) - n + 1)}
    return any(" ".join(q[i : i + n]) in grams for i in range(len(q) - n + 1))


def check_questions(batch: Sequence[ClaimCard], out: GeneratedQuestions) -> list[str]:
    """Rules: one question per claim, a real question, no copied quote.

    Args:
        batch: The claims.
        out: The model's questions.

    Returns:
        Problems; empty if valid.
    """
    wanted = {c.id: c for c in batch}
    got = {q.id: q for q in out.items}
    problems = [f"no question for claim {cid}" for cid in wanted if cid not in got]
    problems += [f"unknown claim id {cid}" for cid in got if cid not in wanted]
    for cid, q in got.items():
        if cid in wanted and not q.question.rstrip().endswith("?"):
            problems.append(f"{cid}: write a question ending with '?'")
        if cid in wanted and _overlap(q.question, wanted[cid].evidence_quote):
            problems.append(f"{cid}: don't copy the quote; ask in your own words")
    return problems


def gold_chunks(card: ClaimCard, chunks: Sequence[Chunk]) -> list[str]:
    """Chunks of the claim's source that contain its quote.

    Args:
        card: Claim card (``id`` prefixed with its source id).
        chunks: The source's text chunks.

    Returns:
        Chunk ids: all that match fully, else the best partial match (straddling quotes),
        else none.
    """
    scored = [(quote_score(card.evidence_quote, c.text), c.id) for c in chunks]
    full = [cid for score, cid in scored if score >= QUOTE_MATCH]
    if full:
        return full
    best = max(scored, default=(0.0, ""))
    return [best[1]] if best[0] >= MIN_GOLD_SCORE else []


def make_cases(s: AskSession, n: int = 60, seed: int = 0) -> list[RetrievalCase]:
    """Sample claim cards, turn them into questions and find their gold chunks.

    Args:
        s: Session (the index holds the claims).
        n: Questions to make (sampled evenly over sources, deterministically).
        seed: Sampling seed.

    Returns:
        Test cases (claims whose quote can't be located are skipped).
    """
    cards = s.index.claims()
    rng = random.Random(seed)
    by_source: dict[str, list[ClaimCard]] = {}
    for card in cards:
        by_source.setdefault(card.id.split(":", 1)[0], []).append(card)
    picked: list[ClaimCard] = []
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(by_source.items())}
    while len(picked) < n and any(pools.values()):
        for pool in pools.values():
            if pool and len(picked) < n:
                picked.append(pool.pop())
    template = load_prompt("questions")
    questions: dict[str, str] = {}
    s.switcher.use(s.llm.model)
    for i in range(0, len(picked), BATCH):
        batch = picked[i : i + BATCH]
        claims = "\n".join(f'{c.id}: {c.claim}\n    quote: "{c.evidence_quote}"' for c in batch)
        out = structured_chat(
            s.llm.backend,
            s.llm.request(template.format(claims=claims)),
            GeneratedQuestions,
            check=partial(check_questions, batch),
        )
        questions |= {q.id: q.question for q in out.items}
    text_chunks = {
        sid: [c for c in s.index.chunks(sid) if c.kind in ("text", "caption", "thesis")]
        for sid in by_source
    }
    cases = []
    for card in picked:
        sid = card.id.split(":", 1)[0]
        gold = gold_chunks(card, text_chunks[sid])
        if gold and card.id in questions:
            cases.append(
                RetrievalCase(
                    claim_id=card.id, source_id=sid, question=questions[card.id], gold=gold
                )
            )
    return cases


def score_method(
    name: str, cases: Sequence[RetrievalCase], ranked: Callable[[RetrievalCase], list[str]]
) -> MethodScore:
    """Recall@k and MRR of one retrieval method.

    Args:
        name: Method name.
        cases: Test cases.
        ranked: Returns the method's top-10 chunk ids for a case.

    Returns:
        The scores.
    """
    hits = {k: 0 for k in KS}
    rr = 0.0
    for case in cases:
        ids = ranked(case)
        first = next((i for i, cid in enumerate(ids, start=1) if cid in case.gold), 0)
        for k in KS:
            hits[k] += bool(first and first <= k)
        rr += 1 / first if first else 0.0
    n = len(cases)
    return MethodScore(
        method=name,
        recall={k: round(hits[k] / n, 3) if n else 0.0 for k in KS},
        mrr=round(rr / n, 3) if n else 0.0,
        n=n,
    )


def evaluate_retrieval(
    s: AskSession, cases: Sequence[RetrievalCase], rewrite: bool = True
) -> list[MethodScore]:
    """Compare BM25, dense, hybrid and hybrid with query rewriting on the test cases.

    Args:
        s: Session.
        cases: Test cases.
        rewrite: Also score hybrid search on model-rewritten queries (one call per case).

    Returns:
        One score per method.
    """
    index: Index = s.index
    tiers = (1, 2)
    vectors = {c.claim_id: s.embedder.query(c.question) for c in cases}

    def run(method: Method) -> Callable[[RetrievalCase], list[str]]:
        def ranked(case: RetrievalCase) -> list[str]:
            hits = index.search(case.question, vectors[case.claim_id], tiers, 10, method)
            return [h.chunk.id for h in hits]

        return ranked

    scores = [score_method(m, cases, run(m)) for m in ("bm25", "dense", "hybrid")]
    if rewrite:
        from labmate.ask.graph import rewrite_query  # noqa: PLC0415 - avoids a cycle

        def rewritten(case: RetrievalCase) -> list[str]:
            query = rewrite_query(s, case.question, "", [])
            hits = index.search(query, s.embedder.query(query), tiers, 10, "hybrid")
            return [h.chunk.id for h in hits]

        s.switcher.use(s.llm.model)
        scores.append(score_method("hybrid + rewrite", cases, rewritten))
    return scores


def retrieval_markdown(scores: Sequence[MethodScore]) -> str:
    """Results table.

    Args:
        scores: One per method.

    Returns:
        Markdown.
    """
    head = "| Method | " + " | ".join(f"Recall@{k}" for k in KS) + " | MRR@10 |"
    rows = [head, "|---|" + "---|" * (len(KS) + 1)]
    for sc in scores:
        cells = " | ".join(f"{sc.recall[k]:.2f}" for k in KS)
        rows.append(f"| {sc.method} | {cells} | {sc.mrr:.2f} |")
    n = scores[0].n if scores else 0
    rows += ["", f"{n} questions generated from verified claim cards; gold = the chunks that "
             "contain the claim's quote."]  # fmt: skip
    return "\n".join(rows) + "\n"


def save_retrieval(
    out: Path, cases: Sequence[RetrievalCase], scores: Sequence[MethodScore]
) -> None:
    """Write ``retrieval.md``, ``retrieval.json`` and the test cases.

    Args:
        out: Output directory.
        cases: Test cases.
        scores: Method scores.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.md").write_text(retrieval_markdown(scores))
    (out / "retrieval.json").write_text(
        json.dumps([s.model_dump() for s in scores], indent=2) + "\n"
    )
    (out / "retrieval_cases.json").write_text(
        json.dumps([c.model_dump() for c in cases], indent=2, ensure_ascii=False) + "\n"
    )


# --- answers --------------------------------------------------------------------------------


class GoldenQuestion(BaseModel):
    """One hand-written test question.

    Attributes:
        question: The question.
        sources: Source ids a good answer cites (any of them); empty = don't check.
        abstain: True if the library does not answer it and the agent should say so.
    """

    question: str
    sources: list[str] = Field(default_factory=list)
    abstain: bool = False


class AnswerCase(BaseModel):
    """How one agent did on one golden question.

    Attributes:
        question: The question.
        agent: ``graph`` or ``prebuilt``.
        abstained: Whether it declined to answer.
        abstain_ok: Declined exactly when it should have.
        source_hit: Cited an expected source (``None`` if not applicable).
        sentences: Cited sentences in the answer.
        supported: Of those, how many the judge found supported by their citations.
        calls: Model calls the answer took.
        seconds: Wall time.
        answer: The answer text.
    """

    question: str
    agent: str
    abstained: bool
    abstain_ok: bool
    source_hit: bool | None
    sentences: int
    supported: int
    calls: int
    seconds: float
    answer: str


def load_golden(path: Path) -> list[GoldenQuestion]:
    """Read the golden set (a YAML list of questions).

    Args:
        path: ``library/golden.yaml``.

    Returns:
        The questions.
    """
    import yaml  # noqa: PLC0415 - only the evaluation needs it

    return [GoldenQuestion.model_validate(q) for q in yaml.safe_load(path.read_text()) or []]


def supported_sentences(s: AskSession, answer: Answer) -> int:
    """Sentences the judge finds supported by their cited chunks (one pass, no rewrites).

    Both agents are scored with the same judge, so the numbers are comparable.

    Args:
        s: Session.
        answer: An answer.

    Returns:
        Number of supported sentences.
    """
    if not answer.sentences:
        return 0
    ids = [c for x in answer.sentences for c in x.chunk_ids]
    loop = judge_pending(
        start_loop(to_blocks(answer.sentences)), evidence_cards(s.index, ids), s.judge, s.workers
    )
    return sum(c.passed for c in loop.report.rounds[-1])


def score_answer(
    s: AskSession, golden: GoldenQuestion, answer: Answer, spans: list[dict[str, object]]
) -> AnswerCase:
    """Score one answer.

    Args:
        s: Session.
        golden: The question and what a good answer looks like.
        answer: The agent's answer.
        spans: Trace spans recorded while answering.

    Returns:
        The case.
    """
    cited = {c.source_id for c in answer.citations}
    root = next((x for x in spans if x.get("name") == "run"), {})
    s.switcher.use(s.judge.model)
    return AnswerCase(
        question=golden.question,
        agent=answer.agent,
        abstained=answer.abstained,
        abstain_ok=answer.abstained == golden.abstain,
        source_hit=bool(cited & set(golden.sources))
        if golden.sources and not golden.abstain
        else None,
        sentences=len(answer.sentences),
        supported=supported_sentences(s, answer),
        calls=sum(x.get("name") == "llm.chat" for x in spans),
        seconds=round(float(str(root.get("latency_ms", 0))) / 1000, 1),
        answer=answer.text,
    )


def answer_markdown(cases: Sequence[AnswerCase]) -> str:
    """Results table per agent.

    Args:
        cases: Scored answers of both agents.

    Returns:
        Markdown.
    """
    rows = [
        "| Agent | Correct answer/refusal | Cited an expected source | Supported sentences "
        "| Model calls (mean) | Time (mean) |",
        "|---|---|---|---|---|---|",
    ]
    for agent in dict.fromkeys(c.agent for c in cases):
        mine = [c for c in cases if c.agent == agent]
        hits = [c.source_hit for c in mine if c.source_hit is not None]
        sentences = sum(c.sentences for c in mine)
        rows.append(
            f"| {agent} | {sum(c.abstain_ok for c in mine)}/{len(mine)} | "
            f"{sum(hits)}/{len(hits)} | "
            f"{sum(c.supported for c in mine)}/{sentences} | "
            f"{sum(c.calls for c in mine) / len(mine):.1f} | "
            f"{sum(c.seconds for c in mine) / len(mine):.0f} s |"
        )
    return "\n".join(rows) + "\n"
