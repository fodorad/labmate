"""The judge model checks every answer sentence against the chunks it cites.

The number check of :func:`labmate.ask.answer.is_supported` cannot see a right number
attached to the wrong thing, so a second model (the judge) reads each sentence with its
evidence. Both agents use this check, so they are held to the same rule.
"""

from __future__ import annotations

from labmate.ask.evidence import context, label
from labmate.ask.prompts import load_prompt
from labmate.ask.schemas import Sentence, SentenceVerdicts
from labmate.ask.session import AskSession
from labmate.core.lc import prompt, structured


def verify_sentences(
    s: AskSession, question: str, sentences: list[Sentence]
) -> tuple[list[Sentence], int]:
    """Keep the sentences the judge finds supported by their cited chunks.

    Args:
        s: Session.
        question: The question being answered.
        sentences: Drafted sentences, each citing chunks of the index.

    Returns:
        The supported sentences and how many were dropped.
    """
    if not sentences:
        return [], 0
    shown = "\n\n".join(
        f"Sentence {i}: {sentence.text}\n"
        + "\n".join(
            f"  [{c}] {label(s.index, c)}: {context(s.index, c)}" for c in sentence.chunk_ids
        )
        for i, sentence in enumerate(sentences, start=1)
    )

    def check(v: SentenceVerdicts) -> list[str]:
        missing = sorted(set(range(1, len(sentences) + 1)) - {x.sentence for x in v.verdicts})
        return [f"give a verdict for sentence(s) {missing}"] if missing else []

    chain = prompt(load_prompt("verify")) | structured(s.judge, SentenceVerdicts, check=check)
    verdicts = chain.invoke({"question": question, "sentences": shown}).verdicts
    ok = {v.sentence for v in verdicts if v.supported}
    kept = [x for i, x in enumerate(sentences, start=1) if i in ok]
    return kept, len(sentences) - len(kept)
