You choose the visual for one slide of a LinkedIn carousel about a research paper.

Slide {position} of {total}: {title}
{bullets}

Figures available from the paper:
{figures}

Call exactly one tool:
- use_paper_figure: if one of the paper's figures directly illustrates this slide.
  Prefer this: it is the most faithful option.
- make_diagram: if no figure fits but the slide describes a process, pipeline, architecture
  or comparison that a small diagram (at most 8 boxes) would make clearer.
- no_visual: for slides that are clearer as text alone, such as the takeaway.

If a tool reports an error, fix the problem and try again, or choose another tool.
