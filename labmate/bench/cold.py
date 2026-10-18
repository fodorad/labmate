"""A cold run: how every benchmark measurement is made.

Each measurement starts with no model loaded and the reply cache off, so it pays what a first
request pays. Before it loads anything the free memory is checked, every model is unloaded when it
ends (also when it fails), and its result is saved at once under a key, so a stopped benchmark can
be run again and continues where it stopped.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from labmate.bench.guard import Memory
from labmate.config import CacheConfig, Config
from labmate.core.ollama import OllamaServer
from labmate.core.usage import UsageCollector

log = logging.getLogger(__name__)

Measure = Callable[[Config, UsageCollector], dict[str, Any]]
"""A measurement: gets a cold configuration and the collector its models report to, returns the
result as JSON-ready data."""


def cold_config(config: Config, collector: UsageCollector) -> Config:
    """A copy of the configuration that reports to ``collector`` and does not use the cache.

    Args:
        config: Loaded configuration.
        collector: Receives every model call.

    Returns:
        The copy.
    """
    return config.model_copy(
        update={"cache": CacheConfig(enabled=False), "callbacks": [collector]}, deep=True
    )


def load_results(file: Path) -> dict[str, dict[str, Any]]:
    """The results saved in a file.

    Args:
        file: A results file written by :class:`ColdRun`.

    Returns:
        Key to result; empty if the file does not exist yet.
    """
    return dict(json.loads(file.read_text())) if file.exists() else {}


@dataclass
class ColdRun:
    """Runs measurements cold and keeps their results in one JSON file.

    Attributes:
        config: Loaded configuration.
        file: The results file (key to result).
        force: Measure keys that already have a result.
        transport: Transport to Ollama (the configured host if omitted; tests).
        memory: The machine's memory (this machine's if omitted).
    """

    config: Config
    file: Path
    force: bool = False
    transport: httpx.BaseTransport | None = None
    memory: Memory = field(default_factory=Memory.here)

    @property
    def server(self) -> OllamaServer:
        """The Ollama server the models run on."""
        return OllamaServer.from_config(self.config, self.transport)

    def results(self) -> dict[str, dict[str, Any]]:
        """The saved results (see :func:`load_results`).

        Returns:
            Key to result.
        """
        return load_results(self.file)

    def run(self, key: str, models_gb: float, measure: Measure) -> None:
        """Measure once, cold, and save the result under ``key``.

        Args:
            key: Names the result, e.g. a model tag.
            models_gb: Size of the models the measurement loads together.
            measure: The measurement.

        Raises:
            NotEnoughMemory: If the models do not fit; nothing was loaded.
        """
        done = self.results()
        if key in done and not self.force:
            log.info("  have %s", key)
            return
        server = self.server
        server.unload_all()
        self.memory.require_room(models_gb)
        log.info("measuring %s", key)
        collector = UsageCollector(server.resident_gb)
        try:
            done[key] = measure(cold_config(self.config, collector), collector)
        finally:
            server.unload_all()
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(done, indent=2) + "\n")
