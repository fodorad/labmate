"""``labmate paper2post <paper>``."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from labmate.config import Config
from labmate.paper2flow.cli import add_paper_arguments, run_paper

if TYPE_CHECKING:
    from labmate.cli import CliContext


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate paper2post``.

    Args:
        sub: The top-level subparsers.
    """
    post = sub.add_parser("paper2post", help="a paper -> runs/<id>/post.pdf")
    add_paper_arguments(post)
    post.set_defaults(handler=run)


def run(config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Turn a paper into ``post.pdf``.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``paper``, ``title``).
        ctx: Transports to the web and to Ollama.

    Returns:
        Exit code (see :func:`labmate.paper2flow.cli.run_paper`).
    """
    from labmate.paper2post.chain import paper2post  # noqa: PLC0415

    return run_paper(paper2post, config, args, ctx)
