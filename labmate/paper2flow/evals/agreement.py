"""Judge agreement: how often does a model's fact-check verdict match a human's?

Each candidate judge re-judges the labelled bullets with the exact prompt the pipeline
uses (:func:`~labmate.core.factcheck.judge_card`), one bullet per call, so the
numbers describe the judge as deployed. Agreement is reported as accuracy and Cohen's
kappa, both on the three labels and collapsed to pass/fail (``supported`` vs the rest),
which is the decision the pipeline actually acts on.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from labmate.core.factcheck import judge_card
from labmate.core.lc import RecordedChatModel, batch_map
from labmate.paper2flow.evals.labels import LabelledBullet
from labmate.paper2flow.schemas import Bullet, Card, ClaimCard, VerdictLabel

LABELS: list[VerdictLabel] = ["supported", "partial", "unsupported"]
"""Label order used in confusion matrices."""


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two raters.

    Args:
        a: Labels of rater A.
        b: Labels of rater B, same length.

    Returns:
        Kappa in [-1, 1]; 1.0 when both raters agree perfectly (even on a single
        class), 0.0 when chance agreement is 1 but they disagree.

    Raises:
        ValueError: On different lengths or empty input.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / n**2
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


class JudgeAgreement(BaseModel):
    """Agreement of one judge model with the human labels.

    Attributes:
        model: Judge model tag.
        n: Labelled bullets compared.
        accuracy: Exact label agreement.
        kappa: Cohen's kappa on the three labels.
        accuracy_binary: Agreement on pass (``supported``) vs fail.
        kappa_binary: Cohen's kappa on pass vs fail.
        confusion: ``confusion[human][model]`` counts.
    """

    model: str
    n: int
    accuracy: float
    kappa: float
    accuracy_binary: float
    kappa_binary: float
    confusion: dict[str, dict[str, int]]


def judge_labels(
    labels: list[LabelledBullet],
    judge: RecordedChatModel,
    workers: int = 2,
    config: RunnableConfig | None = None,
) -> list[VerdictLabel]:
    """Re-judge labelled bullets with one model, using the pipeline's judge prompt.

    Args:
        labels: Bullets with their evidence.
        judge: The judge model.
        workers: Concurrent calls.
        config: Callbacks (tracing).

    Returns:
        One verdict per bullet, in order.
    """

    def one(item: LabelledBullet, cfg: RunnableConfig) -> VerdictLabel:
        cards = {
            f"e{i}": ClaimCard(
                id=f"e{i}",
                claim=quote,
                evidence_quote=quote,
                kind="result",
                section="",
                page=0,
            )
            for i, quote in enumerate(item.evidence, start=1)
        }
        card = Card(title="", bullets=[Bullet(text=item.text, claim_ids=list(cards) or ["e0"])])
        return judge_card(card, cards, judge, cfg).verdicts[0].verdict

    return batch_map(one, labels, config, workers)


def _binary(labels: Sequence[str]) -> list[str]:
    return ["pass" if x == "supported" else "fail" for x in labels]


def agreement(
    human: Sequence[VerdictLabel], predicted: Sequence[VerdictLabel], model: str
) -> JudgeAgreement:
    """Compare a judge's verdicts with the human labels.

    Args:
        human: Human labels.
        predicted: The judge's verdicts, same order.
        model: Judge model tag.

    Returns:
        Accuracy, kappa and the confusion matrix.
    """
    n = len(human)
    confusion = {h: {p: 0 for p in LABELS} for h in LABELS}
    for h, p in zip(human, predicted, strict=True):
        confusion[h][p] += 1
    hb, pb = _binary(human), _binary(predicted)
    return JudgeAgreement(
        model=model,
        n=n,
        accuracy=round(sum(h == p for h, p in zip(human, predicted, strict=True)) / n, 3),
        kappa=round(cohen_kappa(human, predicted), 3),
        accuracy_binary=round(sum(h == p for h, p in zip(hb, pb, strict=True)) / n, 3),
        kappa_binary=round(cohen_kappa(hb, pb), 3),
        confusion=confusion,
    )


def agreement_markdown(results: list[JudgeAgreement]) -> str:
    """Agreement table plus one confusion matrix per judge.

    Args:
        results: One entry per judge model.

    Returns:
        Markdown.
    """
    lines = [
        "| Judge | n | Accuracy | Cohen's κ | Pass/fail accuracy | Pass/fail κ |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.model}` | {r.n} | {r.accuracy:.2f} | {r.kappa:.2f} | "
            f"{r.accuracy_binary:.2f} | {r.kappa_binary:.2f} |"
        )
    for r in results:
        lines += [
            "",
            f"**`{r.model}`** (rows: human, columns: judge)",
            "",
            "| | " + " | ".join(LABELS) + " |",
            "|---|" + "---|" * len(LABELS),
        ]
        lines += [
            f"| {h} | " + " | ".join(str(r.confusion[h][p]) for p in LABELS) + " |" for h in LABELS
        ]
    return "\n".join(lines) + "\n"
