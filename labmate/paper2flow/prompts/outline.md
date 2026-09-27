You plan a one-glance overview of a research paper for ML engineers and researchers.
The overview has exactly four blocks, like the project page of a research portfolio:

1. task: what the paper sets out to do: the problem, its inputs and outputs, the setting
   and why it matters.
2. challenges: what makes the problem hard, and where earlier approaches fall short.
3. method: what the authors propose and how it works (for a benchmark: how it is built;
   for a survey: how it organises the field).
4. results: the main results with their numbers and comparisons, plus notable ablations
   or limitations.

Paper: {title}
Paper type: {paper_type}

Rules:
- Return the four blocks in this order, with purposes "task", "challenges", "method" and
  "results".
- Build every block on 3 to 5 claim cards from the list below, referenced by id, so it has
  material for specific bullets. Do not use a claim in more than two blocks.
- Match claims to blocks by their kind: [task] and [contribution] claims for the task,
  [challenge] claims for the challenges, [method] claims for the method, [result] and
  [limitation] claims for the results. Background about other approaches belongs to the
  challenges, not the task.
- Prefer claims with concrete numbers, named datasets, baselines and components. Put
  claims whose numbers can be compared (a result and its baselines) in the results block.
- "title": a specific headline for the block (at most 8 words), not just the block name.
- "hook": a concrete, curiosity-raising one-line summary of the paper, not its title, at
  most 10 words.

Claim cards:
{claims}
