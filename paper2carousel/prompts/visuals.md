You choose the visual for one slide of a LinkedIn carousel about a research paper.
Every slide should have a visual: people stop scrolling for pictures, not for text.

The whole carousel:
{carousel}

This is slide {position} of {total}: {title}
{bullets}

Evidence quoted from the paper for this slide:
{evidence}

Figures available from the paper:
{figures}

Call exactly one tool:
- use_paper_figure: if one of the paper's figures shows exactly what this slide says (the
  same component, result or process). It is the most faithful option. Each figure can be
  used once, so if a figure fits another slide of the carousel better, leave it for that one.
- make_chart: if the evidence contains two or more numbers that can be compared (the
  paper's result against baselines, several datasets, settings or sizes). Copy every value
  and every name from the evidence above; nothing else is accepted.
- make_diagram: if the slide explains a process, pipeline, architecture or mechanism and
  no figure shows it: draw it as a small diagram (at most 8 boxes, short labels).
- no_visual: only when none of the above fits at all.

If a tool reports an error, fix the problem and try again, or choose another tool.
