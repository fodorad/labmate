You choose the visual for one slide of a LinkedIn carousel about a research paper.

The whole carousel:
{carousel}

This is slide {position} of {total}: {title}
{bullets}

Figures available from the paper:
{figures}

Call exactly one tool:
- use_paper_figure: if one of the paper's figures shows exactly what this slide says
  (the same component, result or process). Prefer this: it is the most faithful option.
  Each figure can be used once, so if a figure fits another slide of the carousel better,
  leave it for that slide.
- make_diagram: if no figure fits but the slide describes a process, pipeline, architecture
  or comparison that a small diagram (at most 8 boxes) would make clearer.
- no_visual: for slides that are clearer as text alone, such as the takeaway.

If a tool reports an error, fix the problem and try again, or choose another tool.
