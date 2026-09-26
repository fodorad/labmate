Draw the proposed method of a research paper as a pipeline graph for a LinkedIn post
image: the inputs on the left, the method's components and processing steps in the
middle, the outputs on the right.

Paper: {title}

- 4 to 10 nodes. "kind": input (what goes in), data (datasets or intermediate
  representations), component (a model part, e.g. an encoder or attention module),
  process (an operation, e.g. fusion or training on a dataset union) or output.
- "label": at most 5 words, using the names the paper uses (they are checked against the
  evidence below; a label naming something the evidence doesn't is rejected).
- Edges follow the data flow; every node has at least one edge. Edge labels are optional
  (at most 3 words).
- "caption": one sentence describing the flow.

What the method slide says:
{bullets}

Evidence quoted from the paper:
{evidence}
