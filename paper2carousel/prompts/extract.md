You extract claim cards from one section of a research paper.

A claim card is one self-contained statement a reader should remember: a contribution,
how the method works, a result, or a limitation. For each claim:
- "claim": one sentence in your own words, understandable without the paper.
- "evidence_quote": a short passage copied character for character from the SECTION TEXT
  below (at most 30 words). It must support the claim. Do not paraphrase, fix typos or
  merge sentences in the quote.
- Keep every number exactly as written.

Return at most {max_claims} claims, the most important first. Return an empty list if the
section has nothing worth a slide (for example acknowledgements).

Paper: {title}
Section: {section} (page {page})

SECTION TEXT:
{text}
