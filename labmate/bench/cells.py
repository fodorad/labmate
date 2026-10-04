"""A benchmark cell (one use case on one model profile) and its tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

TARGETS_S = {
    "ask: graph": 30.0,
    "ask: agent": 30.0,
    "cv2job": 60.0,
    "scout": 180.0,
    "paper2flow": 300.0,
    "paper2post": 300.0,
}
"""Seconds a run should take to count as fast enough (per question for ask)."""


@dataclass
class Cell:
    """One use case run on one profile.

    Attributes:
        use_case: ``ask: graph``, ``ask: agent``, ``cv2job``, ``scout``, ``paper2flow`` or
            ``paper2post``.
        profile: Model profile name.
        writer: Writer model.
        judge: Judge model.
        seconds: Wall time (per question for ask).
        calls: Model calls.
        tokens_in: Tokens read.
        tokens_out: Tokens written.
        load_s: Seconds spent loading models.
        swaps: Models loaded again after being pushed out.
        peak_gb: Most memory loaded at once.
        quality: Score in percent.
        parts: The quality parts, between 0 and 1.
        error: What went wrong, if the run failed.
    """

    use_case: str
    profile: str
    writer: str
    judge: str
    seconds: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    load_s: float = 0.0
    swaps: int = 0
    peak_gb: float = 0.0
    quality: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)
    error: str = ""


def save(cell: Cell, directory: Path) -> Path:
    """Write a cell as JSON.

    Args:
        cell: The cell.
        directory: ``evals/bench/cells``.

    Returns:
        The file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{cell.use_case.replace(': ', '-')}__{cell.profile}.json"
    file = directory / name
    file.write_text(json.dumps(asdict(cell), indent=2) + "\n")
    return file


def load(directory: Path) -> list[Cell]:
    """Read every saved cell.

    Args:
        directory: ``evals/bench/cells``.

    Returns:
        The cells, in file-name order.
    """
    return [Cell(**json.loads(f.read_text())) for f in sorted(directory.glob("*.json"))]


def verdict(cell: Cell) -> str:
    """Whether a run was fast enough.

    Args:
        cell: A finished cell.

    Returns:
        ``fast`` if within the target for its use case, else ``slow``.
    """
    return "fast" if cell.seconds <= TARGETS_S.get(cell.use_case, float("inf")) else "slow"


def cells_markdown(cells: list[Cell]) -> str:
    """One table per use case, fastest profile first.

    Args:
        cells: The cells.

    Returns:
        Markdown.
    """
    parts = []
    for use_case in dict.fromkeys(c.use_case for c in cells):
        mine = sorted((c for c in cells if c.use_case == use_case), key=lambda c: c.seconds)
        target = TARGETS_S.get(use_case)
        per = " per question" if use_case.startswith("ask") else ""
        parts += [
            f"### {use_case}", "",
            f"Target: under {target:.0f} s{per}. Quality parts are between 0 and 1.", "",
            "| Profile | Writer | Judge | Time | Fast enough | Loading | Swaps | Peak memory "
            "| Calls | Tokens in/out | Quality | Parts |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]  # fmt: skip
        for c in mine:
            if c.error:
                parts.append(
                    f"| {c.profile} | `{c.writer}` | `{c.judge}` | failed: {c.error}" + " |" * 8
                )
                continue
            detail = ", ".join(f"{k} {v:.2f}" for k, v in c.parts.items())
            parts.append(
                f"| {c.profile} | `{c.writer}` | `{c.judge}` | {c.seconds:.0f} s "
                f"| {'yes' if verdict(c) == 'fast' else '**no**'} | {c.load_s:.0f} s | {c.swaps} "
                f"| {c.peak_gb:.1f} GB | {c.calls} | {c.tokens_in}/{c.tokens_out} "
                f"| **{c.quality:.0f}%** | {detail} |"
            )
        parts.append("")
    return "\n".join(parts)
