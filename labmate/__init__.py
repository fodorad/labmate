"""labmate: local research tools on Ollama.

Features:

- :mod:`labmate.paper2flow`: a paper in, a fact-checked overview with data-flow diagrams
  out (a chain).
- :mod:`labmate.paper2post`: a paper in, a fact-checked LinkedIn post out (a chain).
- :mod:`labmate.scout`: a topic in, notes on arXiv papers out (an agent).
- :mod:`labmate.cv2job`: a CV and a job posting in, a tailored CV, a cover letter and a gap
  report out (a chain with one agent step).
- :mod:`labmate.ask`: questions about a library of PDFs, answered with citations by a
  LangGraph graph or a tool-calling agent.

All run on :mod:`labmate.core`: local models via Ollama with a reply cache.
"""

__version__ = "0.0.0"
"""Package version; kept in step with ``pyproject.toml``."""
