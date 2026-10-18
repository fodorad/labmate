"""``labmate cv2job <cv> <job>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from labmate.config import Config

if TYPE_CHECKING:
    from labmate.cli import CliContext


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate cv2job``.

    Args:
        sub: The top-level subparsers.
    """
    cv = sub.add_parser("cv2job", help="a CV and a job posting -> tailored CV, letter, gap report")
    cv.add_argument("cv", type=Path, help="your CV as YAML (see examples/cv2job/cv.yaml)")
    cv.add_argument("job", type=Path, help="the job posting as a text file")
    cv.add_argument("--out", type=Path, help="output folder (default: runs/cv2job/<job file>)")
    cv.set_defaults(handler=run)


def run(config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Tailor a CV to a job posting.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``cv``, ``job``, ``out``).
        ctx: The transport to Ollama and how the gap agent asks the candidate a question.

    Returns:
        Exit code: 0 on success, 1 if an input file is missing or invalid.
    """
    from labmate.cv2job.chain import cv2job  # noqa: PLC0415 - needs the [ask] extra

    out = args.out or config.tracing.runs_dir / "cv2job" / args.job.stem
    try:
        pdfs = cv2job(config, args.cv, args.job, out, ctx.ask, ollama=ctx.ollama)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for pdf in pdfs:
        print(pdf)
    return 0
