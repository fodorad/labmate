"""Compare triage deciders on a labelled set: accuracy of the action and time per decision.

``evals/triage/golden.yaml`` holds real arXiv papers (title and abstract) and what a reader with
each of two sets of interests should do with them: ``deep``, ``post`` or ``skip``. The labels were
written by hand. Each decider rates every paper against every interest; a decision model and a
chat model are held to the same labels.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from labmate.bench.micro import unload_all
from labmate.config import CacheConfig, Config
from labmate.core.ingest import ArxivMetadata
from labmate.triage.decide import decide, is_decision_model
from labmate.triage.systemone import DecisionError

GOLDEN = Path("evals/triage/golden.yaml")
"""The labelled papers."""


@dataclass
class DeciderResult:
    """How one decider did on the labelled set.

    Attributes:
        model: Model tag.
        kind: ``decision`` or ``chat``.
        cases: Number of paper and interest pairs.
        exact: Cases where the action equals the label.
        read: Cases where "read it (deep or post) or skip it" equals the label's.
        deep: Cases where "deep or not" equals the label's.
        median_s: Median seconds per decision (the first one pays the model load).
        error: What went wrong, if the model could not be used.
    """

    model: str
    kind: str = ""
    cases: int = 0
    exact: int = 0
    read: int = 0
    deep: int = 0
    median_s: float = 0.0
    error: str = ""


def load_golden(path: Path = GOLDEN) -> dict[str, Any]:
    """Read the labelled set.

    Args:
        path: The YAML file.

    Returns:
        ``{"interests": {key: text}, "papers": [{id, title, abstract, expected: {key: action}}]}``.
    """
    return dict(yaml.safe_load(path.read_text()))


def evaluate(
    config: Config,
    model: str,
    golden: dict[str, Any],
    transport: httpx.BaseTransport | None = None,
) -> DeciderResult:
    """Run one decider over every labelled case.

    Args:
        config: Loaded configuration.
        model: The decider model tag.
        golden: The labelled set (see :func:`load_golden`).
        transport: Transport to Ollama (the configured host if omitted; tests).

    Returns:
        The scores; ``error`` is set if the model could not answer.
    """
    cfg = config.model_copy(deep=True)
    cfg.cache, cfg.models.decider = CacheConfig(enabled=False), model
    result = DeciderResult(model)
    unload_all(cfg.ollama.host, transport)
    times: list[float] = []
    try:
        decision_model = is_decision_model(cfg, transport)
        result.kind = "decision" if decision_model else "chat"
        for key, interests in golden["interests"].items():
            for paper in golden["papers"]:
                meta = ArxivMetadata(title=paper["title"], authors=[], abstract=paper["abstract"])
                started = time.perf_counter()
                action = decide(
                    cfg, meta, interests, decision_model=decision_model, transport=transport
                ).action
                times.append(time.perf_counter() - started)
                label = paper["expected"][key]
                result.cases += 1
                result.exact += action == label
                result.read += (action != "skip") == (label != "skip")
                result.deep += (action == "deep") == (label == "deep")
    except (httpx.HTTPError, DecisionError, ValueError, RuntimeError) as e:
        result.error = f"{type(e).__name__}: {e}"[:160]
    finally:
        unload_all(cfg.ollama.host, transport)
    result.median_s = statistics.median(times) if times else 0.0
    return result


def triage_markdown(results: list[DeciderResult], sizes: dict[str, float]) -> str:
    """The deciders as a table, best first.

    Args:
        results: One entry per decider.
        sizes: Model tag to gigabytes.

    Returns:
        Markdown.
    """
    rows = [
        "| Decider | Kind | Size | Action right | Read or skip right | Deep or not right "
        "| Median per paper |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r.exact):
        if r.error:
            rows.append(f"| `{r.model}` | {r.kind} | | failed: {r.error} | | | |")
            continue
        rows.append(
            f"| `{r.model}` | {r.kind} | {sizes.get(r.model, 0.0):.1f} GB "
            f"| {r.exact}/{r.cases} | {r.read}/{r.cases} | {r.deep}/{r.cases} "
            f"| {r.median_s:.1f} s |"
        )
    return "\n".join(rows) + "\n"


def to_dict(result: DeciderResult) -> dict[str, Any]:
    """A result as JSON-ready data."""
    return asdict(result)
