"""Statements to judge against their evidence: some right, some wrong in ways code cannot see.

The pipeline already catches a number that is not in the evidence. These cases are harder: every
number and name is in the evidence, but the statement puts it on the wrong thing (a layer count read
as a head count, a task swapped), or claims something the evidence never says. Only a judge model
can catch those. A good judge calls the clean statements ``supported`` and the others not.
"""

from __future__ import annotations

from dataclasses import dataclass

TRANSFORMER = (
    "The encoder is composed of a stack of N = 6 identical layers. The base model uses h = 8 "
    "parallel attention heads with d_model = 512 and d_k = d_v = 64."
)
TRANSLATION = (
    "The big Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, "
    "improving over the existing best results by over 2 BLEU."
)
DATA = (
    "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million "
    "sentence pairs."
)
SPEED = (
    "Self-attention layers are faster than recurrent layers when the sequence length n is smaller "
    "than the representation dimensionality d, which is most often the case with sentence "
    "representations used by state-of-the-art models in machine translation."
)
OPTIMIZER = (
    "We used the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 10^-9. We varied the "
    "learning rate over the course of training: increasing it linearly for the first warmup_steps "
    "training steps, and decreasing it thereafter proportionally to the inverse square root of the "
    "step number. We used warmup_steps = 4000."
)
REGULARIZATION = (
    "We apply dropout to the output of each sub-layer. For the base model, we use a rate of "
    "P_drop = 0.1. During training, we employed label smoothing of value 0.1. This hurts "
    "perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."
)


@dataclass(frozen=True)
class JudgeCase:
    """One statement and what a good judge says about it.

    Attributes:
        evidence: The quote the statement cites.
        statement: The statement.
        right: True if the evidence supports it.
        kind: What kind of mistake it is, for the report.
    """

    evidence: str
    statement: str
    right: bool
    kind: str


def _clean(evidence: str, statement: str) -> JudgeCase:
    return JudgeCase(evidence, statement, True, "clean")


def _wrong(evidence: str, statement: str, kind: str) -> JudgeCase:
    return JudgeCase(evidence, statement, False, kind)


CASES = [
    _clean(TRANSFORMER, "The base model uses 8 attention heads."),
    _clean(TRANSFORMER, "Each attention head has a dimension of 64."),
    _clean(TRANSFORMER, "The encoder stacks 6 identical layers."),
    _clean(TRANSLATION, "The big model reaches 28.4 BLEU on English-to-German."),
    _clean(DATA, "Training used about 4.5 million English-German sentence pairs."),
    _wrong(TRANSFORMER, "The base model uses 6 attention heads.", "number on the wrong thing"),
    _wrong(TRANSFORMER, "The encoder stacks 8 identical layers.", "numbers swapped"),
    _wrong(TRANSFORMER, "The model dimension is 64 and each head has 512.", "numbers swapped"),
    _wrong(TRANSLATION, "The big model reaches 28.4 BLEU on English-to-French.", "task swapped"),
    _wrong(TRANSLATION, "The big model was trained on 8 GPUs for 3.5 days.", "not in the evidence"),
    _wrong(DATA, "The sentences were filtered with a language classifier.", "not in the evidence"),
    _wrong(DATA, "Training used the English-French dataset of 4.5 million pairs.", "task swapped"),
    _clean(SPEED, "Self-attention is faster than recurrence when n is smaller than d."),
    _clean(OPTIMIZER, "The learning rate rises linearly for the first 4000 steps."),
    _clean(REGULARIZATION, "Label smoothing hurts perplexity but improves BLEU."),
    _clean(
        REGULARIZATION, "Dropout is applied to the output of every sub-layer, at a rate of 0.1."
    ),
    _wrong(
        SPEED, "Self-attention layers are always faster than recurrent layers.", "overgeneralized"
    ),
    _wrong(SPEED, "Self-attention is faster when n is larger than d.", "comparison reversed"),
    _wrong(OPTIMIZER, "After warmup the learning rate decreases linearly.", "wrong shape"),
    _wrong(OPTIMIZER, "The warmup lasts 4000 epochs.", "unit swapped"),
    _wrong(OPTIMIZER, "The Adam optimizer uses beta2 = 0.9.", "number on the wrong thing"),
    _wrong(REGULARIZATION, "Label smoothing improves perplexity and BLEU.", "comparison reversed"),
    _wrong(REGULARIZATION, "Label smoothing was used to prevent overfitting.", "cause not stated"),
    _wrong(REGULARIZATION, "Dropout is applied only to the final layer.", "scope changed"),
]
"""Nine clean statements and fifteen mistakes."""
