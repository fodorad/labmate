"""``labmate scout <topic>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from labmate.config import Config

if TYPE_CHECKING:
    from labmate.cli import CliContext


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate scout``.

    Args:
        sub: The top-level subparsers.
    """
    scout = sub.add_parser("scout", help="an agent that searches arXiv and writes notes on a topic")
    scout.add_argument("topic", help="what to look into, e.g. 'efficient attention for video'")
    scout.add_argument("--deep", type=int, default=2, help="papers it may look at deeply (slow)")
    scout.add_argument("--out", type=Path, help="output folder (default: runs/scout/<topic>)")
    scout.set_defaults(handler=run)


def run(config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Let the scout agent look into a topic.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``topic``, ``deep``, ``out``).
        ctx: Transports to arXiv and to Ollama.

    Returns:
        Exit code: 0 if notes were written, 1 if the agent stopped without them.
    """
    from labmate.core.ingest import slugify  # noqa: PLC0415
    from labmate.scout.agent import run_scout  # noqa: PLC0415 - needs the [ask] extra

    out = args.out or config.tracing.runs_dir / "scout" / slugify(args.topic)
    notes = run_scout(config, args.topic, out, args.deep, ctx.web, ctx.ollama)
    if notes is None:
        print("error: the agent stopped without writing notes", file=sys.stderr)
        return 1
    print(notes)
    return 0
