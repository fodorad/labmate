Answer a question about a library of research documents, using only the evidence below.

Question: {question}

Rules:
- Write 1 to 8 short sentences (at most 35 words each). Each sentence lists the ids of
  the evidence chunks it is based on in "chunk_ids"; do not write ids in the text.
- Keep names and numbers exactly as in the evidence. Say only what the evidence says:
  if it answers the question only partly, answer that part and say what is not covered.
- Prefer the passages that answer the question most directly. Tables appear as flattened
  rows of numbers whose column names are far away: never read a number off such a row;
  take it from a sentence that names what the number is.

Evidence:
{evidence}
