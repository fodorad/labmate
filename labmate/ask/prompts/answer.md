Answer a question about a researcher's work, using only the evidence below.

Question: {question}

Rules:
- The dissertation (tier 1) is the source of truth. Prefer it; use the papers (tier 2)
  for details the dissertation does not give; use outside papers (tier 3) only for
  related work.
- Where sources disagree (listed below), state the dissertation's value and mention the
  other value as the one reported in that source.
- Write 1 to 8 short sentences (at most 35 words each). Each sentence lists the ids of
  the evidence chunks it is based on in "chunk_ids"; do not write ids in the text.
- Keep names and numbers exactly as in the evidence. Say only what the evidence says:
  if it answers the question only partly, answer that part and say what is not covered.
- Write in the third person about the researcher ("the dissertation proposes ...").

Disagreements between sources:
{conflicts}

Evidence:
{evidence}
