"""Guards that stop a benchmark before it can exhaust the machine's memory.

Loading models that do not fit makes macOS swap and can kill other programs, so the benchmark
checks the free memory before it loads anything and unloads every model when it stops.
"""

from __future__ import annotations

import re
import subprocess

from labmate.bench.report import GPU_SHARE, HEADROOM_GB

PAGE = re.compile(r"page size of (\d+) bytes")
"""The page size line of ``vm_stat``."""

COUNTED = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
"""The ``vm_stat`` counters that macOS can hand out without swapping."""


class NotEnoughMemory(RuntimeError):
    """The models would not fit in the memory that is free."""


def parse_vm_stat(text: str) -> float:
    """Free memory from the output of ``vm_stat``.

    Args:
        text: What ``vm_stat`` printed.

    Returns:
        Gigabytes of memory that can be handed out without swapping.
    """
    found = PAGE.search(text)
    size = int(found.group(1)) if found else 16384
    pages = 0
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name.strip() in COUNTED:
            pages += int(value.strip().rstrip("."))
    return pages * size / 1e9


def free_memory_gb() -> float:
    """The memory that is free now.

    Returns:
        Gigabytes, from ``vm_stat``.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout  # noqa: S607
    return parse_vm_stat(out)


def require_room(models_gb: float, ram_gb: float, free_gb: float | None = None) -> None:
    """Refuse to load models that do not fit.

    Args:
        models_gb: Size of the models to load together.
        ram_gb: The machine's memory.
        free_gb: Memory that is free now (measured if omitted).

    Raises:
        NotEnoughMemory: If they exceed what the GPU may use (less the headroom), or what is
            free now (plus the headroom).
    """
    limit = GPU_SHARE * ram_gb - HEADROOM_GB
    if models_gb > limit:
        raise NotEnoughMemory(
            f"{models_gb:.1f} GB of models do not fit in {limit:.1f} GB "
            f"({GPU_SHARE:.0%} of {ram_gb:.0f} GB, keeping {HEADROOM_GB:.0f} GB free)"
        )
    free = free_memory_gb() if free_gb is None else free_gb
    if models_gb + HEADROOM_GB > free:
        raise NotEnoughMemory(
            f"only {free:.1f} GB are free, {models_gb + HEADROOM_GB:.1f} GB needed: "
            "close other programs or run fewer models"
        )
