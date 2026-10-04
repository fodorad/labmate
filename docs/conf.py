"""Sphinx configuration for labmate."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "labmate"
copyright = "2026, Ádám Fodor"
author = "Ádám Fodor"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",  # allows index.md alongside README.md
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,  # surfaces module-level attrs/consts if documented
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

templates_path = ["_templates"]
# graphs.md and pipelines.md hold Mermaid diagrams, which GitHub renders
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "graphs.md", "pipelines.md"]

html_theme = "furo"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Render "Attributes:" sections as :ivar: fields so they don't clash with autodoc's
# attribute entries for pydantic fields (duplicate object description warnings).
napoleon_use_ivar = True
# Mermaid blocks (rendered by GitHub) are shown as plain code here.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference", "misc.highlighting_failure"]

# LangChain builds its pydantic models lazily on first attribute access; when that happens
# inside autodoc the schema generation fails, so load everything the docs touch up front.
import labmate.ask.studio  # noqa: E402,F401
import labmate.cv2job.chain  # noqa: E402,F401
import labmate.paper2post.chain  # noqa: E402,F401
import labmate.scout.agent  # noqa: E402,F401
import labmate.triage.run  # noqa: E402,F401


def _drop_foreign_docstrings(app, what, name, obj, options, lines):
    """Blank the docstrings of names a module only imports.

    Autodoc skips them anyway, but only after the type-hint extension has parsed their
    docstrings, and some third-party docstrings are not valid reStructuredText.
    """
    module = getattr(obj, "__module__", None)
    if what != "module" and isinstance(module, str) and not module.startswith("labmate"):
        lines[:] = []


def setup(app):
    app.connect("autodoc-process-docstring", _drop_foreign_docstrings, priority=100)
