"""Command-line interface: ``labmate <command>``; every command is also a Makefile target.

- ``paper2flow`` / ``paper2post``: a paper in, ``overview.pdf`` / ``post.pdf`` out.
- ``ask``: questions about your research (see :mod:`labmate.ask.cli`).
- ``eval``: metrics of the finished paper runs.
- ``graphs``: Mermaid diagrams of the chains and graphs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
from ollama import ResponseError

from labmate.config import Config, load_config
from labmate.core.ingest import IngestError
from labmate.paper2flow.chain import ARTIFACTS, NothingSupportedError, paper2flow
from labmate.paper2flow.evals.metrics import results_markdown, run_metrics
from labmate.paper2post.chain import paper2post


def cmd_paper(
    config: Config,
    args: argparse.Namespace,
    web: httpx.BaseTransport | None,
    ollama: httpx.BaseTransport | None,
) -> int:
    """Run paper2flow or paper2post for one paper.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``command``, ``paper``, ``title``).
        web: Transport for downloads (tests); the network otherwise.
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code: 0 on success, 1 if the paper can't be read or nothing survives the
        fact-check.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    chain = paper2flow if args.command == "paper2flow" else paper2post
    try:
        out = chain(config, args.paper, args.title, web, ollama)
    except (IngestError, NothingSupportedError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{'Output:':8}{out}")
    return 0


def finished_runs(config: Config, run_dirs: Sequence[Path]) -> list[Path]:
    """The given run directories, or every run that reached the fact-check.

    Args:
        config: Loaded configuration (for ``runs_dir``).
        run_dirs: Explicit run directories; empty means "all".

    Returns:
        Run directories containing the fact-check artifact, sorted.
    """
    candidates = list(run_dirs) or sorted(config.tracing.runs_dir.glob("*"))
    return [d for d in candidates if (d / ARTIFACTS["checked"]).exists()]


def cmd_eval(config: Config, run_dirs: Sequence[Path], out: Path) -> int:
    """Compute run metrics and write ``results.md`` / ``results.json``.

    Args:
        config: Loaded configuration.
        run_dirs: Runs to evaluate (default: all finished runs).
        out: Output directory.

    Returns:
        Exit code: 0 on success, 1 if there is no finished run.
    """
    runs = finished_runs(config, run_dirs)
    if not runs:
        print(f"error: no finished runs (need {ARTIFACTS['checked']})", file=sys.stderr)
        return 1
    metrics = [run_metrics(d) for d in runs]
    out.mkdir(parents=True, exist_ok=True)
    table = results_markdown(metrics)
    (out / "results.md").write_text(table)
    (out / "results.json").write_text(
        json.dumps([m.model_dump() for m in metrics], indent=2) + "\n"
    )
    print(table)
    print(f"Wrote {out / 'results.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser: one command per feature, plus the shared commands.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="labmate", description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, output in (("paper2flow", "overview.pdf"), ("paper2post", "post.pdf")):
        chain = sub.add_parser(name, help=f"a paper -> runs/<id>/{output}")
        chain.add_argument("paper", help="arXiv id or URL, PDF URL or PDF path")
        chain.add_argument("--title", help="title override for PDFs without a usable title")

    from labmate.ask.cli import add_parser as add_ask  # noqa: PLC0415 - light imports only

    add_ask(sub)

    eval_p = sub.add_parser("eval", help="metrics of the finished paper runs -> evals/results.md")
    eval_p.add_argument("runs", nargs="*", type=Path, help="run dirs (default: all)")
    eval_p.add_argument("--out", type=Path, default=Path("evals"), help="output directory")

    graphs = sub.add_parser("graphs", help="Mermaid diagrams of the chains and graphs -> docs/")
    graphs.add_argument("--out", type=Path, default=Path("docs/graphs.md"))
    return parser


def main(
    argv: Sequence[str] | None = None,
    web: httpx.BaseTransport | None = None,
    ollama: httpx.BaseTransport | None = None,
) -> int:
    """Entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).
        web: Injected web transport (tests); the network otherwise.
        ollama: Injected transport to Ollama (tests); the configured host otherwise.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "graphs":
        from labmate.diagrams import write_diagrams  # noqa: PLC0415 - needs the [ask] extra

        print(f"Wrote {write_diagrams(args.out)}")
        return 0
    if args.command == "eval":
        return cmd_eval(config, args.runs, args.out)
    try:
        if args.command == "ask":
            from labmate.ask.cli import main as ask_main  # noqa: PLC0415 - needs the [ask] extra

            return ask_main(config, args, ollama)
        return cmd_paper(config, args, web, ollama)
    except (httpx.TransportError, ResponseError) as e:  # Ollama down, or the model not pulled
        print(
            f"error: {e} (is Ollama running at {config.ollama.host}, with the models?)",
            file=sys.stderr,
        )
        return 2
