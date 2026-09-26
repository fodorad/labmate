You extract claim cards from one section of a research paper.

A claim card is one self-contained statement a reader should remember. Its "kind":
- task: what the paper sets out to do (the problem, its inputs and outputs, the setting).
- challenge: what makes the problem hard, or where earlier approaches fall short.
- contribution: what the authors claim as new.
- method: how the proposed approach works.
- result: a measured outcome, ideally with numbers and the comparison.
- limitation: where the approach falls short.
For each claim:
- "claim": one sentence in your own words, understandable without the paper. It must not
  say more than the quote: every name, number and cause in the claim is in the quote.
- "evidence_quote": a passage copied character for character from the SECTION TEXT below
  (one or two full sentences, at most 45 words). It must support the whole claim on its
  own, including what a number belongs to: the model, dataset, metric or setting. Do not
  paraphrase, fix typos or merge sentences in the quote.
- Keep every number exactly as written.
- For results, prefer quotes that contain the numbers, including the comparison: the
  paper's result and the baseline or previous best. A row of a results table, copied as
  it appears in the text below (e.g. "Transformer (big) 28.4 41.8"), is a valid quote.

Return at most {max_claims} claims, the most important first. Return an empty list if the
section has nothing worth a slide (for example acknowledgements).

Paper: {title}
Section: {section} (page {page})

SECTION TEXT:
{text}
