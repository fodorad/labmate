Draw the end-to-end flow of a research paper as a top-to-bottom flow diagram: {goal}.

Paper: {title}

- 5 to 10 boxes. "kind": input (the raw data that goes in), data (datasets or
  intermediate representations, e.g. features or embeddings), component (a model part,
  e.g. an encoder or an attention module), process (an operation, e.g. preprocessing,
  fusion or training) or output (the target output).
- The flow starts at the raw input data (kind input or data) and ends at the target
  output (kind output). Every box is on the path between them.
- "label": at most 5 words, using the names the paper uses (labels are checked against
  the evidence below; a label naming something the evidence doesn't is rejected). No
  numbers that are not in the evidence.
- Arrows follow the data, in reading order. An arrow label says what flows along it
  (e.g. "frame sequence", "landmark features"; at most 4 words) or is "".
- "title": at most 8 words. "caption": one or two sentences walking through the flow.
- "expand": the ids of 1 to 3 component or process boxes whose inner steps the evidence
  describes in enough detail for a diagram of their own.

What the overview says about the method:
{bullets}

Evidence from the paper:
{evidence}
