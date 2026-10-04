"""Markdown tables of the benchmark results."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

from labmate.bench.micro import ModelResult
from labmate.config import Config

HEADROOM_GB = 5.0
"""Memory kept for macOS and other programs when asking whether models fit."""


def memory_gb() -> float:
    """The machine's memory in gigabytes.

    Returns:
        Physical memory, in gigabytes (1e9 bytes, as Ollama counts).
    """
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def machine() -> str:
    """A one-line description of the machine for the report header.

    Returns:
        Chip, memory and operating system.
    """
    system = f"{platform.system()} {platform.release()}"
    return f"{platform.machine()}, {memory_gb():.0f} GB memory, {system}"


@dataclass(frozen=True)
class ProfileRow:
    """A profile and whether its models fit in memory together.

    Attributes:
        profile: Profile name.
        writer: Writer model.
        judge: Judge model.
        gb: Gigabytes the models take together.
        fits: Whether that leaves :data:`HEADROOM_GB` free.
    """

    profile: str
    writer: str
    judge: str
    gb: float
    fits: bool


def profile_rows(config: Config, sizes: dict[str, float], ram_gb: float) -> list[ProfileRow]:
    """Which profiles fit in memory together.

    Args:
        config: Configuration with its ``[profiles]``.
        sizes: Model tag to gigabytes.
        ram_gb: The machine's memory.

    Returns:
        One row per profile: writer, judge, gigabytes together and whether that fits.
    """
    rows = []
    for name, profile in config.profiles.items():
        together = sum(sizes.get(tag, 0.0) for tag in dict.fromkeys([profile.text, profile.critic]))
        rows.append(
            ProfileRow(
                name, profile.text, profile.critic, together, together <= ram_gb - HEADROOM_GB
            )
        )
    return rows


def micro_markdown(results: list[ModelResult], config: Config, ram_gb: float) -> str:
    """The micro benchmark as tables.

    Args:
        results: One entry per model.
        config: Configuration with its ``[profiles]``.
        ram_gb: The machine's memory.

    Returns:
        Markdown: one table of models, one of profiles.
    """
    models = [
        "| Model | Size | Cold load | Reads | Writes | Structured | Tool calls "
        "| Judge: mistakes caught | Judge: false alarms | Judge: s per verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r.size_gb):
        if r.error:
            models.append(f"| `{r.model}` | {r.size_gb:.1f} GB | failed: {r.error} | | | | | | | |")
            continue
        j = r.judge
        models.append(
            f"| `{r.model}` | {r.size_gb:.1f} GB | {r.load_s:.0f} s | {r.prompt_tps:.0f} tok/s "
            f"| {r.decode_tps:.0f} tok/s | {r.structured_ok}/5 "
            f"| {'-' if r.tools_ok is None else f'{r.tools_ok}/3'} "
            f"| {'-' if j is None else f'{j.caught}/{j.mistakes}'} "
            f"| {'-' if j is None else f'{j.false_alarms}/{j.clean}'} "
            f"| {'-' if j is None else f'{j.seconds_per_verdict:.1f} s'} |"
        )
    sizes = {r.model: r.size_gb for r in results}
    profiles = [
        f"| Profile | Writer | Judge | Together "
        f"| Fits in {ram_gb:.0f} GB (keeping {HEADROOM_GB:.0f} GB free) |",
        "|---|---|---|---|---|",
    ]
    for row in profile_rows(config, sizes, ram_gb):
        profiles.append(
            f"| {row.profile} | `{row.writer}` | `{row.judge}` | {row.gb:.1f} GB "
            f"| {'yes' if row.fits else '**no, models are swapped**'} |"
        )
    return "\n".join(["### Models", "", *models, "", "### Profiles", "", *profiles, ""])
