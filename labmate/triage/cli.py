"""``labmate triage <topic>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from labmate.config import Config

if TYPE_CHECKING:
    from labmate.cli import CliContext


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate triage``.

    Args:
        sub: The top-level subparsers.
    """
    triage = sub.add_parser("triage", help="a small model sorts arXiv hits: deep read, post, skip")
    triage.add_argument("topic", help="what to search for, e.g. 'efficient attention for video'")
    triage.add_argument(
        "--interests",
        type=Path,
        default=Path("private/interests.md"),
        help="a text file on what you work on (default: private/interests.md)",
    )
    triage.add_argument("--interest", help="the interests as text, instead of a file")
    triage.add_argument("--limit", type=int, default=8, help="papers to look at")
    triage.add_argument("--budget", type=int, default=2, help="most papers to read deeply")
    triage.add_argument("--run", action="store_true", help="also make the overviews and posts")
    triage.add_argument("--out", type=Path, help="output folder (default: runs/triage/<topic>)")
    triage.set_defaults(handler=run)


def run(config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Let the decider model sort the arXiv hits for a topic.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``topic``, ``interests``, ``interest``, ``limit``, ``budget``,
            ``run``, ``out``).
        ctx: Transports to arXiv and to Ollama.

    Returns:
        Exit code: 0 on success, 1 if there are no interests to judge by.
    """
    from labmate.core.ingest import slugify  # noqa: PLC0415
    from labmate.triage.run import run_triage  # noqa: PLC0415 - needs the [ask] extra

    if args.interest:
        interests = args.interest
    elif args.interests.exists():
        interests = args.interests.read_text().strip()
    else:
        print(f"error: say what you work on: --interest TEXT or the file {args.interests}",
              file=sys.stderr)  # fmt: skip
        return 1
    out = args.out or config.tracing.runs_dir / "triage" / slugify(args.topic)
    run_triage(
        config, args.topic, interests, out, args.limit, args.budget, args.run, ctx.web, ctx.ollama
    )
    print(out / "triage.md")
    return 0
