"""Model phases: keep only one large model in memory at a time.

On a 32 GB Mac the writer (21 GB) and the critic (16 GB) can't be resident together.
Steps call :meth:`ModelSwitcher.use` before each batch of calls for a model; switching
unloads the previous model explicitly and counts the swap, so the cost of alternating
shows up in the trace instead of as a mysterious slowdown.
"""

from __future__ import annotations

import logging

from paper2flow.llm.client import OllamaClient

log = logging.getLogger(__name__)


class ModelSwitcher:
    """Tracks the active model and unloads the previous one on a switch.

    Args:
        client: Live Ollama client, or ``None`` in replay mode (switching is then a no-op
            apart from bookkeeping, so replayed traces still show the phases).
        unload_wait_s: How long to wait for memory to be freed.
    """

    def __init__(self, client: OllamaClient | None, unload_wait_s: float = 30.0) -> None:
        self.client = client
        self.unload_wait_s = unload_wait_s
        self.active: str | None = None
        self.swaps = 0

    def use(self, model: str) -> None:
        """Make ``model`` the active one, unloading the previous model if it differs.

        Args:
            model: Model tag about to be used.
        """
        if model == self.active:
            return
        if self.active is not None:
            self.swaps += 1
            log.info("  phase: %s -> %s", self.active, model)
            if self.client is not None:
                self.client.unload(self.active, wait_s=self.unload_wait_s)
        self.active = model

    def release(self) -> None:
        """Unload the active model (end of run)."""
        if self.active is not None and self.client is not None:
            self.client.unload(self.active)
        self.active = None
