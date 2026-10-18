"""Guards that stop a benchmark before it can exhaust the machine's memory.

Loading models that do not fit makes the machine swap and can kill other programs, so the
benchmark checks the free memory before it loads anything and unloads every model when it stops.
The free memory comes from ``vm_stat`` on macOS and from ``/proc/meminfo`` on Linux.
"""

from __future__ import annotations

import platform
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from labmate.bench.report import HEADROOM_GB, MEMORY_SHARE, memory_gb

PAGE = re.compile(r"page size of (\d+) bytes")
"""The page size line of ``vm_stat``."""

COUNTED = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
"""The ``vm_stat`` counters that macOS can hand out without swapping."""


MEMINFO = Path("/proc/meminfo")
"""Where Linux reports its memory."""


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


def parse_meminfo(text: str) -> float:
    """Free memory from Linux's ``/proc/meminfo``.

    Args:
        text: The file's content.

    Returns:
        Gigabytes the kernel can hand out without swapping (``MemAvailable``).

    Raises:
        ValueError: If there is no ``MemAvailable`` line.
    """
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name == "MemAvailable":
            return int(value.split()[0]) * 1024 / 1e9
    raise ValueError("no MemAvailable in /proc/meminfo")


def free_memory_gb(system: str | None = None) -> float:
    """The memory that is free now.

    Args:
        system: ``platform.system()`` (the running one if omitted).

    Returns:
        Gigabytes, from ``vm_stat`` on macOS and ``/proc/meminfo`` on Linux.

    Raises:
        NotEnoughMemory: On another system, where free memory cannot be read.
    """
    system = system or platform.system()
    if system == "Darwin":
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout  # noqa: S607
        return parse_vm_stat(out)
    if system == "Linux":
        return parse_meminfo(MEMINFO.read_text())
    raise NotEnoughMemory(f"cannot read the free memory on {system}")


def require_room(
    models_gb: float, ram_gb: float, free_gb: float | None = None, share: float = MEMORY_SHARE
) -> None:
    """Refuse to load models that do not fit.

    Args:
        models_gb: Size of the models to load together.
        ram_gb: The machine's memory.
        free_gb: Memory that is free now (measured if omitted).
        share: The share of ``ram_gb`` models may use (see
            :func:`labmate.bench.report.memory_share`).

    Raises:
        NotEnoughMemory: If they exceed what the GPU may use (less the headroom), or what is
            free now (plus the headroom).
    """
    limit = share * ram_gb - HEADROOM_GB
    if models_gb > limit:
        raise NotEnoughMemory(
            f"{models_gb:.1f} GB of models do not fit in {limit:.1f} GB "
            f"({share:.0%} of {ram_gb:.0f} GB, keeping {HEADROOM_GB:.0f} GB free)"
        )
    free = free_memory_gb() if free_gb is None else free_gb
    if models_gb + HEADROOM_GB > free:
        raise NotEnoughMemory(
            f"only {free:.1f} GB are free, {models_gb + HEADROOM_GB:.1f} GB needed: "
            "close other programs or run fewer models"
        )


@dataclass(frozen=True)
class Memory:
    """The machine's memory, as the benchmark sees it.

    Attributes:
        ram_gb: Physical memory.
        free_gb: Returns the memory that is free now.
        share: The share of ``ram_gb`` models may use (see
            :func:`labmate.bench.report.memory_share`).
    """

    ram_gb: float
    free_gb: Callable[[], float]
    share: float = MEMORY_SHARE

    @classmethod
    def here(cls) -> Memory:
        """This machine's memory.

        Returns:
            Its physical memory, a probe of its free memory and its share.
        """
        return cls(memory_gb(), free_memory_gb)

    def require_room(self, models_gb: float) -> None:
        """Refuse to load models that do not fit (see :func:`require_room`).

        Args:
            models_gb: Size of the models to load together.
        """
        require_room(models_gb, self.ram_gb, self.free_gb(), self.share)
