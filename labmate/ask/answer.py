"""Cited answers: the checks every sentence must pass and the numbered citations.

Shared by the graph and the agent, so both are held to the same rule: a sentence stays
only if the chunks it cites exist and contain every number the sentence states.
"""

from __future__ import annotations

from labmate.ask.evidence import context, label
from labmate.ask.index import Index
from labmate.ask.schemas import Answer, Citation, Sentence
from labmate.core.factcheck import numbers_in

MAX_SENTENCE_WORDS = 35
"""Longest answer sentence."""

OFF_TOPIC = "That question is outside the library, so I can't answer it from sources."
NOT_FOUND = "The library doesn't answer that, as far as I can find."


def structure_problems(sentences: list[Sentence], known: set[str]) -> list[str]:
    """Rules a draft must follow before it is checked against the evidence.

    Args:
        sentences: The draft's sentences.
        known: Chunk ids the writer was shown.

    Returns:
        Problems to show the model; empty if the draft is well formed.
    """
    problems = []
    for i, sentence in enumerate(sentences, start=1):
        unknown = sorted(set(sentence.chunk_ids) - known)
        if unknown:
            problems.append(f"sentence {i} cites ids that are not in the evidence: {unknown}")
        if len(sentence.text.split()) > MAX_SENTENCE_WORDS:
            problems.append(f"sentence {i} has more than {MAX_SENTENCE_WORDS} words")
        if "[" in sentence.text:
            problems.append(f"sentence {i} has ids in its text; list them in chunk_ids")
    return problems


def is_supported(index: Index, sentence: Sentence) -> bool:
    """Whether the cited chunks exist and state every number of the sentence.

    Args:
        index: The index.
        sentence: One answer sentence.

    Returns:
        True if every cited chunk exists and holds all the sentence's numbers.
    """
    try:
        evidence = " ".join(context(index, cid) for cid in sentence.chunk_ids)
    except KeyError:
        return False
    return numbers_in(sentence.text) <= numbers_in(evidence)


def compose(
    index: Index, question: str, sentences: list[Sentence], dropped: int = 0, agent: str = "graph"
) -> Answer:
    """Keep the supported sentences, number their citations and assemble the answer.

    Args:
        index: The index.
        question: The question as asked.
        sentences: The draft's sentences.
        dropped: Sentences already removed before this step.
        agent: Which agent answered.

    Returns:
        The answer; an abstention if no sentence is supported.
    """
    kept = [s for s in sentences if is_supported(index, s)]
    dropped += len(sentences) - len(kept)
    if not kept:
        return Answer(question=question, text=NOT_FOUND, abstained=True,
                      reason="no supported sentence", dropped=dropped, agent=agent)  # fmt: skip
    numbers: dict[str, int] = {}
    parts = []
    for sentence in kept:
        marks = []
        for cid in sentence.chunk_ids:
            numbers.setdefault(cid, len(numbers) + 1)
            marks.append(f"[{numbers[cid]}]")
        parts.append(f"{sentence.text.rstrip()} {''.join(marks)}")
    citations = []
    for cid, n in numbers.items():
        chunk = index.chunk(cid)
        citations.append(
            Citation(n=n, chunk_id=cid, source_id=chunk.source_id, label=label(index, cid),
                     text=chunk.text)
        )  # fmt: skip
    return Answer(question=question, text=" ".join(parts), sentences=kept, citations=citations,
                  dropped=dropped, agent=agent)  # fmt: skip
