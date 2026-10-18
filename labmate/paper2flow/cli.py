"""``labmate paper2flow <paper>`` and ``labmate eval``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import httpx

from labmate.config import Config

if TYPE_CHECKING:
    from labmate.cli import CliContext


class PaperChain(Protocol):
    """``paper2flow`` or ``paper2post``: one paper in, the path of its PDF out."""

    def __call__(
        self,
        config: Config,
        source: str,
        title: str | None = None,
        web: httpx.BaseTransport | None = None,
        ollama: httpx.BaseTransport | None = None,
    ) -> Path:
        """Turn ``source`` into a PDF (see :func:`labmate.paper2flow.chain.paper2flow`)."""
        ...


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate paper2flow`` and ``labmate eval``.

    Args:
        sub: The top-level subparsers.
    """
    flow = sub.add_parser("paper2flow", help="a paper -> runs/<id>/overview.pdf")
    add_paper_arguments(flow)
    flow.set_defaults(handler=run)

    evals = sub.add_parser("eval", help="metrics of the finished paper runs -> evals/results.md")
    evals.add_argument("runs", nargs="*", type=Path, help="run dirs (default: all)")
    evals.add_argument("--out", type=Path, default=Path("evals"), help="output directory")
    evals.set_defaults(handler=run_eval)


def add_paper_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments of a command that turns one paper into a PDF.

    Args:
        parser: The command's parser.
    """
    parser.add_argument("paper", help="arXiv id or URL, PDF URL or PDF path")
    parser.add_argument("--title", help="title override for PDFs without a usable title")


def run_paper(chain: PaperChain, config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Turn one paper into a PDF with ``chain``.

    Args:
        chain: ``paper2flow`` or ``paper2post``.
        config: Loaded configuration.
        args: Parsed arguments (``paper``, ``title``).
        ctx: Transports to the web and to Ollama.

    Returns:
        Exit code: 0 on success, 1 if the paper can't be read or nothing survives the
        fact-check.
    """
    from labmate.core.ingest import IngestError  # noqa: PLC0415
    from labmate.paper2flow.chain import NothingSupportedError  # noqa: PLC0415

    try:
        out = chain(config, args.paper, args.title, ctx.web, ctx.ollama)
    except (IngestError, NothingSupportedError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{'Output:':8}{out}")
    return 0


def run(config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Turn a paper into ``overview.pdf``.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``paper``, ``title``).
        ctx: Transports to the web and to Ollama.

    Returns:
        Exit code (see :func:`run_paper`).
    """
    from labmate.paper2flow.chain import paper2flow  # noqa: PLC0415

    return run_paper(paper2flow, config, args, ctx)


def finished_runs(config: Config, run_dirs: Sequence[Path]) -> list[Path]:
    """The given run directories, or every run that reached the fact-check.

    Args:
        config: Loaded configuration (for ``runs_dir``).
        run_dirs: Explicit run directories; empty means "all".

    Returns:
        Run directories containing the fact-check artifact, sorted.
    """
    from labmate.paper2flow.chain import ARTIFACTS  # noqa: PLC0415

    candidates = list(run_dirs) or sorted(config.tracing.runs_dir.glob("*"))
    return [d for d in candidates if (d / ARTIFACTS["checked"]).exists()]


def run_eval(config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Compute run metrics and write ``results.md`` / ``results.json``.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``runs``: runs to evaluate, default all finished; ``out``).
        ctx: Unused.

    Returns:
        Exit code: 0 on success, 1 if there is no finished run.
    """
    from labmate.paper2flow.chain import ARTIFACTS  # noqa: PLC0415
    from labmate.paper2flow.evals.metrics import results_markdown, run_metrics  # noqa: PLC0415

    runs = finished_runs(config, args.runs)
    if not runs:
        print(f"error: no finished runs (need {ARTIFACTS['checked']})", file=sys.stderr)
        return 1
    metrics = [run_metrics(d) for d in runs]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    table = results_markdown(metrics)
    (out / "results.md").write_text(table)
    (out / "results.json").write_text(
        json.dumps([m.model_dump() for m in metrics], indent=2) + "\n"
    )
    print(table)
    print(f"Wrote {out / 'results.md'}")
    return 0
