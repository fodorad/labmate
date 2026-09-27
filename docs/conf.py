"""Sphinx configuration for paper2flow."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "paper2flow"
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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Render "Attributes:" sections as :ivar: fields so they don't clash with autodoc's
# attribute entries for pydantic fields (duplicate object description warnings).
napoleon_use_ivar = True
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]

# LangGraph's dependency langchain-core does not import cleanly under autodoc's type
# hint processing; mocking it is enough to document the engine module.
autodoc_mock_imports = ["langgraph", "langchain_core"]
