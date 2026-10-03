You write one card of a one-glance overview of a research paper.

This is the "{label}" card of a four-card overview (task, challenges, proposed method,
main results): {purpose_hint}

Write a title (at most 8 words, specific to this paper) and 3 or 4 bullets (at most 25 words
each). Rules:
- Be specific: every bullet carries at least one concrete detail from the evidence, such
  as a number, a dataset, a baseline, a component or a named comparison. No generic
  statements like "achieves strong results".
- Use only facts from the claim cards below. Each bullet lists the ids of the claims it uses.
- Keep every number exactly as it appears in the evidence.
- Base each bullet on the evidence quote. The claim text is a paraphrase and can say more
  than the quote; anything that is only in the paraphrase will fail the fact-check.
- Write for a reader who has not read the paper: third person ("the authors"), plain text,
  no Markdown, no emoji, no hashtags.

Claim cards for this card:
{claims}
