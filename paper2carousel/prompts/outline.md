You plan a LinkedIn carousel about a research paper for ML engineers and researchers.

Paper: {title}
Paper type: {paper_type}

Plan {n_min} to {n_max} slides. Allowed slide purposes, in the recommended order:
{purposes}

Rules:
- The first slide's purpose is "{first}"; the last slide's purpose is "takeaway".
- Every slide is built on 2 to 4 claim cards from the list below, referenced by id, so it
  has enough material for specific bullets. The takeaway may use 1 or 2.
- Prefer claims with concrete numbers, named datasets, baselines and components. Put
  claims whose numbers can be compared (a result and its baselines) on the same slide.
- Do not use a claim on more than two slides.
- "hook" is the cover headline: a concrete, curiosity-raising statement, not the paper
  title, at most 10 words.

Claim cards:
{claims}
