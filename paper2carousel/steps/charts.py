"""Bar charts drawn from numbers in the paper, as SVG (no plotting library needed).

The visuals agent asks for a chart with labels and values; :func:`check_chart` only
accepts values that appear in the slide's verified evidence quotes, so a generated chart
can't show a number the paper doesn't state. The SVG is deterministic (same input, same
bytes), which keeps replayed runs byte-identical.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape

from paper2carousel.steps.factcheck import numbers_in
from paper2carousel.steps.render import Theme

MIN_BARS = 2
"""A chart needs something to compare."""

MAX_BARS = 8
"""More bars than this don't read on a phone."""

WIDTH = 1000
"""SVG width in user units (the slide scales it to fit)."""

ROW = 76
"""Height of one bar row."""

MAX_LABEL = 24
"""Longest bar label that fits the label column."""

_GENERIC = frozenset(
    "the and for with from model models method methods score scores result results new all "
    "our their this that previous previously best state art".split()
)
"""Words that don't identify what a number belongs to."""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def _number(value: object) -> float | None:
    try:
        return float(str(value).replace(",", "").strip().rstrip("%"))
    except ValueError:
        return None


def check_chart(
    labels: list[str], values: list[object], quotes: list[str]
) -> tuple[list[str], list[str]]:
    """Validate a chart request against the slide's evidence quotes.

    Each bar must be backed by one quote that contains both its value and a word of its
    label, so a number can't be attached to the wrong thing (e.g. an English-French score
    labelled as a previous model's English-German score).

    Args:
        labels: Bar labels.
        values: Bar values (numbers or numeric strings).
        quotes: The slide's evidence quotes.

    Returns:
        ``(problems, shown)``: problems (empty if the chart is valid) and each value
        written exactly as in the evidence (``41.0`` stays ``41.0``, not ``41``).
    """
    problems: list[str] = []
    if not MIN_BARS <= len(labels) <= MAX_BARS:
        problems.append(f"use {MIN_BARS} to {MAX_BARS} bars, not {len(labels)}")
    if len(labels) != len(values):
        problems.append(f"{len(labels)} labels but {len(values)} values")
    parsed = [(q, _tokens(q) - _GENERIC, {n: _number(n) for n in numbers_in(q)}) for q in quotes]
    shown: list[str] = []
    for label, value in zip(labels, values, strict=False):
        if len(label) > MAX_LABEL:
            problems.append(f"label {label!r} is longer than {MAX_LABEL} characters")
        number = _number(value)
        with_value = [
            (words, next(s for s, v in nums.items() if v == number))
            for _, words, nums in parsed
            if number is not None and number in nums.values()
        ]
        if not with_value:
            problems.append(f"value {value} for {label!r} is not in the evidence")
            continue
        backed = [text for words, text in with_value if _tokens(label) - _GENERIC & words]
        if not backed:
            problems.append(
                f"no evidence quote gives {value} for {label!r}; pair each value with the "
                "name it belongs to in the same quote"
            )
            continue
        shown.append(backed[0])
    return problems, shown


def bar_chart_svg(
    labels: list[str],
    shown: list[str],
    unit: str,
    highlight: str = "",
    theme: Theme | None = None,
) -> str:
    """Horizontal bar chart in the slide palette; bars start at zero.

    Args:
        labels: Bar labels.
        shown: Values as written in the paper (see :func:`check_chart`).
        unit: Unit or metric name shown above the bars (e.g. ``BLEU``).
        highlight: Label of the bar to emphasise (the paper's own method).
        theme: Colours.

    Returns:
        SVG source.
    """
    theme = theme or Theme()
    values = [_number(v) or 0.0 for v in shown]
    top = max(max(values), 1e-9)
    label_w, bar_x0, bar_w = 330, 350, 520
    height = 56 + ROW * len(labels)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="{escape(theme.font)}, sans-serif">',
        f'<text x="{bar_x0}" y="30" font-size="24" font-weight="600" '
        f'fill="{theme.muted}">{escape(unit)}</text>',
    ]
    for i, (label, value, text) in enumerate(zip(labels, values, shown, strict=True)):
        y = 56 + i * ROW
        width = max(bar_w * value / top, 3)
        emphasis = highlight and label.strip().lower() == highlight.strip().lower()
        fill = theme.accent if emphasis else theme.muted
        opacity = "1" if emphasis else "0.45"
        weight = "700" if emphasis else "400"
        rows += [
            f'<text x="{label_w}" y="{y + 33}" font-size="26" text-anchor="end" '
            f'font-weight="{weight}" fill="{theme.text}">{escape(label)}</text>',
            f'<rect x="{bar_x0}" y="{y + 8}" width="{width:.1f}" height="40" rx="6" '
            f'fill="{fill}" fill-opacity="{opacity}"/>',
            f'<text x="{bar_x0 + width + 12:.1f}" y="{y + 37}" font-size="28" '
            f'font-weight="700" fill="{theme.text}">{escape(text)}</text>',
        ]
    rows.append("</svg>")
    return "\n".join(rows) + "\n"
