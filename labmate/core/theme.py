"""The shared look: adamfodor.com colours and the bundled Inter font."""

from __future__ import annotations

from importlib.resources import files

from pydantic import BaseModel

FONTS_DIR = files("labmate.core.templates").joinpath("fonts")
"""Bundled fonts (Inter, SIL OFL 1.1), so renders don't depend on the machine's fonts."""


class Theme(BaseModel):
    """Colours (hex), font and handle. Defaults follow the adamfodor.com light theme."""

    background: str = "#f7f6f4"
    surface: str = "#fffefd"
    border: str = "#dcd8d2"
    text: str = "#222b35"
    muted: str = "#4c6176"
    accent: str = "#ab4c31"
    accent_muted: str = "#f1e1dc"
    font: str = "Inter"
    handle: str = "adamfodor.com"
