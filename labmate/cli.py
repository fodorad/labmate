"""Command-line interface: ``labmate <command>``; every command is also a Makefile target.

- ``paper2flow`` / ``paper2post``: a paper in, ``overview.pdf`` / ``post.pdf`` out.
- ``eval``: metrics of the finished paper runs.
- ``scout``: a research agent that searches arXiv and writes notes on a topic.
- ``triage``: a small decision model sorts arXiv hits into deep read, post or skip.
- ``cv2job``: a CV and a job posting in, a tailored CV, a cover letter and a gap report out.
- ``ask``: questions about your research (see :mod:`labmate.ask.cli`).
- ``bench``: measure models and use cases (speed, memory, quality); not part of ``make check``.
- ``graphs``: Mermaid diagrams of the chains and graphs.

Each use case registers its own commands: its package has a ``cli`` module whose ``add_parser``
adds them and sets the function that runs them (``handler``). Adding a use case means adding its
package to :data:`FEATURES`.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from ollama import ResponseError

from labmate.config import Config, load_config
from labmate.core.ollama import DecisionError

if TYPE_CHECKING:
    from labmate.bench.guard import Memory

FEATURES = ("paper2flow", "paper2post", "scout", "triage", "cv2job", "ask", "bench")
"""The packages that register commands (``labmate.<name>.cli``), in help order."""


def terminal_ask(question: str) -> str:
    """Ask the candidate in the terminal; no answer when there is no terminal to ask in.

    Args:
        question: The question.

    Returns:
        The typed answer, or ``""``.
    """
    if not sys.stdin.isatty():
        return ""
    print(f"\n{question}")  # pragma: no cover - interactive
    return input("> ")  # pragma: no cover


@dataclass(frozen=True)
class CliContext:
    """What a command gets besides the configuration and its arguments.

    Attributes:
        web: Transport for arXiv and downloads (tests); the network if ``None``.
        ollama: Transport to the Ollama server (tests); the configured host if ``None``.
        ask: How cv2job's agent asks the candidate a question.
        memory: The machine's memory as the benchmark sees it (tests); measured if ``None``.
    """

    web: httpx.BaseTransport | None = None
    ollama: httpx.BaseTransport | None = None
    ask: Callable[[str], str] = terminal_ask
    memory: Memory | None = None


def run_graphs(config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Write the Mermaid diagrams of the chains and graphs.

    Args:
        config: Loaded configuration (unused).
        args: Parsed arguments (``out``).
        ctx: Unused.

    Returns:
        Exit code 0.
    """
    from labmate.diagrams import write_diagrams  # noqa: PLC0415 - needs the [ask] extra

    print(f"Wrote {write_diagrams(args.out)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser: the commands of every feature, plus ``graphs``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="labmate", description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    parser.add_argument("--profile", help="model profile from [profiles] (or LABMATE_PROFILE)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in FEATURES:
        importlib.import_module(f"labmate.{name}.cli").add_parser(sub)
    graphs = sub.add_parser("graphs", help="Mermaid diagrams of the chains and graphs -> docs/")
    graphs.add_argument("--out", type=Path, default=Path("docs/graphs.md"))
    graphs.set_defaults(handler=run_graphs)
    return parser


def main(
    argv: Sequence[str] | None = None,
    web: httpx.BaseTransport | None = None,
    ollama: httpx.BaseTransport | None = None,
    ask: Callable[[str], str] = terminal_ask,
    memory: Memory | None = None,
) -> int:
    """Entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).
        web: Injected web transport (tests); the network otherwise.
        ollama: Injected transport to Ollama (tests); the configured host otherwise.
        ask: How cv2job's agent asks you a question (default: in the terminal).
        memory: The machine's memory for ``bench`` (tests); measured otherwise.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, args.profile)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return int(args.handler(config, args, CliContext(web, ollama, ask, memory)))
    except httpx.TransportError as e:
        print(f"error: {e} (is Ollama running at {config.ollama.host}?)", file=sys.stderr)
        return 2
    except DecisionError as e:  # a decision model not pulled, or refused
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ResponseError as e:  # a model not pulled, or a reply Ollama could not parse
        print(f"error: Ollama refused the request: {e}. Pulled the models? Run again.",
              file=sys.stderr)  # fmt: skip
        return 2
