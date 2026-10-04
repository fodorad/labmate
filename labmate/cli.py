"""Command-line interface: ``labmate <command>``; every command is also a Makefile target.

- ``paper2flow`` / ``paper2post``: a paper in, ``overview.pdf`` / ``post.pdf`` out.
- ``bench``: measure models and use cases (speed, memory, quality); not part of ``make check``.
- ``scout``: a research agent that searches arXiv and writes notes on a topic.
- ``triage``: a small decision model sorts arXiv hits into deep read, post or skip.
- ``cv2job``: a CV and a job posting in, a tailored CV, a cover letter and a gap report out.
- ``ask``: questions about your research (see :mod:`labmate.ask.cli`).
- ``eval``: metrics of the finished paper runs.
- ``graphs``: Mermaid diagrams of the chains and graphs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
from ollama import ResponseError

from labmate.config import Config, load_config
from labmate.core.ingest import IngestError, slugify
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
    parser.add_argument("--profile", help="model profile from [profiles] (or LABMATE_PROFILE)")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, output in (("paper2flow", "overview.pdf"), ("paper2post", "post.pdf")):
        chain = sub.add_parser(name, help=f"a paper -> runs/<id>/{output}")
        chain.add_argument("paper", help="arXiv id or URL, PDF URL or PDF path")
        chain.add_argument("--title", help="title override for PDFs without a usable title")

    scout = sub.add_parser("scout", help="an agent that searches arXiv and writes notes on a topic")
    scout.add_argument("topic", help="what to look into, e.g. 'efficient attention for video'")
    scout.add_argument("--deep", type=int, default=2, help="papers it may look at deeply (slow)")
    scout.add_argument("--out", type=Path, help="output folder (default: runs/scout/<topic>)")

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

    cv = sub.add_parser("cv2job", help="a CV and a job posting -> tailored CV, letter, gap report")
    cv.add_argument("cv", type=Path, help="your CV as YAML (see examples/cv2job/cv.yaml)")
    cv.add_argument("job", type=Path, help="the job posting as a text file")
    cv.add_argument("--out", type=Path, help="output folder (default: runs/cv2job/<job file>)")

    from labmate.ask.cli import add_parser as add_ask  # noqa: PLC0415 - light imports only
    from labmate.bench.cli import add_parser as add_bench  # noqa: PLC0415

    add_ask(sub)
    add_bench(sub)

    eval_p = sub.add_parser("eval", help="metrics of the finished paper runs -> evals/results.md")
    eval_p.add_argument("runs", nargs="*", type=Path, help="run dirs (default: all)")
    eval_p.add_argument("--out", type=Path, default=Path("evals"), help="output directory")

    graphs = sub.add_parser("graphs", help="Mermaid diagrams of the chains and graphs -> docs/")
    graphs.add_argument("--out", type=Path, default=Path("docs/graphs.md"))
    return parser


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


def cmd_scout(
    config: Config,
    args: argparse.Namespace,
    web: httpx.BaseTransport | None,
    ollama: httpx.BaseTransport | None,
) -> int:
    """Let the scout agent look into a topic.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``topic``, ``deep``, ``out``).
        web: Transport for arXiv (tests); the network otherwise.
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code: 0 if notes were written, 1 if the agent stopped without them.
    """
    from labmate.scout.agent import run_scout  # noqa: PLC0415 - needs the [ask] extra

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = args.out or config.tracing.runs_dir / "scout" / slugify(args.topic)
    notes = run_scout(config, args.topic, out, args.deep, web, ollama)
    if notes is None:
        print("error: the agent stopped without writing notes", file=sys.stderr)
        return 1
    print(notes)
    return 0


def cmd_triage(
    config: Config,
    args: argparse.Namespace,
    web: httpx.BaseTransport | None,
    ollama: httpx.BaseTransport | None,
) -> int:
    """Let the decision model sort the arXiv hits for a topic.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``topic``, ``interests``, ``interest``, ``limit``, ``budget``,
            ``run``, ``out``).
        web: Transport for arXiv (tests); the network otherwise.
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code: 0 on success, 1 if there are no interests to judge by.
    """
    from labmate.triage.run import run_triage  # noqa: PLC0415 - needs the [ask] extra

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.interest:
        interests = args.interest
    elif args.interests.exists():
        interests = args.interests.read_text().strip()
    else:
        print(f"error: say what you work on: --interest TEXT or the file {args.interests}",
              file=sys.stderr)  # fmt: skip
        return 1
    out = args.out or config.tracing.runs_dir / "triage" / slugify(args.topic)
    run_triage(config, args.topic, interests, out, args.limit, args.budget, args.run, web, ollama)
    print(out / "triage.md")
    return 0


def cmd_cv2job(
    config: Config,
    args: argparse.Namespace,
    ask: Callable[[str], str],
    ollama: httpx.BaseTransport | None,
) -> int:
    """Tailor a CV to a job posting.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``cv``, ``job``, ``out``).
        ask: Asks the candidate a question (the gap agent uses it).
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code: 0 on success, 1 if an input file is missing or invalid.
    """
    from labmate.cv2job.chain import cv2job  # noqa: PLC0415 - needs the [ask] extra

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = args.out or config.tracing.runs_dir / "cv2job" / args.job.stem
    try:
        pdfs = cv2job(config, args.cv, args.job, out, ask, ollama=ollama)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for pdf in pdfs:
        print(pdf)
    return 0


def main(
    argv: Sequence[str] | None = None,
    web: httpx.BaseTransport | None = None,
    ollama: httpx.BaseTransport | None = None,
    ask: Callable[[str], str] = terminal_ask,
) -> int:
    """Entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).
        web: Injected web transport (tests); the network otherwise.
        ollama: Injected transport to Ollama (tests); the configured host otherwise.
        ask: How cv2job's agent asks you a question (default: in the terminal).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, args.profile)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
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
        if args.command == "bench":
            from labmate.bench.cli import main as bench_main  # noqa: PLC0415

            return bench_main(config, args)
        if args.command == "scout":
            return cmd_scout(config, args, web, ollama)
        if args.command == "triage":
            return cmd_triage(config, args, web, ollama)
        if args.command == "cv2job":
            return cmd_cv2job(config, args, ask, ollama)
        return cmd_paper(config, args, web, ollama)
    except httpx.TransportError as e:
        print(f"error: {e} (is Ollama running at {config.ollama.host}?)", file=sys.stderr)
        return 2
    except ResponseError as e:  # a model not pulled, or a reply Ollama could not parse
        print(f"error: Ollama refused the request: {e}. Pulled the models? Run again.",
              file=sys.stderr)  # fmt: skip
        return 2
