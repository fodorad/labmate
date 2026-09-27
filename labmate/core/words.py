"""Topic words of a text: the crude but deterministic matching the checks rely on."""

from __future__ import annotations

import re

_STOPWORDS = frozenset(
    "about above after also among another based because been being between both could does "
    "each from have into its more most other over same show shows shown such than that the "
    "their them then there these they this those through under using used uses very were "
    "what when where which while with within without would case figure".split()
)


def content_words(
    text: str, ignore: frozenset[str] = frozenset(), split_hyphens: bool = False
) -> set[str]:
    """Topic words of a text: lower-cased words of 4+ letters, crude plural stripping.

    Hyphenated words are joined ("multi-modal" matches "multimodal"), so a compound counts
    once. With ``split_hyphens`` their parts are added too, which is how the paper title's
    words are collected ("Transformer-Based" yields "transformer").

    Args:
        text: Any text.
        ignore: Words to leave out (e.g. the paper title's words).
        split_hyphens: Also include the parts of hyphenated words.

    Returns:
        The set of content words.
    """
    text = text.lower()
    raw = re.findall(r"[a-z]{4,}", re.sub(r"(?<=[a-z])-(?=[a-z])", "", text))
    if split_hyphens:
        raw += re.findall(r"[a-z]{4,}", text)
    words = {w[:-1] if len(w) > 4 and w.endswith("s") else w for w in raw}
    return words - _STOPWORDS - ignore
