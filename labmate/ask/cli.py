"""``labmate ask ...`` commands (imported only when used: they need the ``[ask]`` extra)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from labmate.config import Config

if TYPE_CHECKING:
    from labmate.ask.schemas import Answer
    from labmate.ask.session import AskSession


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate ask <action>``.

    Args:
        sub: The top-level subparsers.
    """
    ask = sub.add_parser("ask", help="questions about a library of documents, with citations")
    ask.add_argument("--library", type=Path, help="library folder (default: [ask].library)")
    actions = ask.add_subparsers(dest="action", required=True)

    index = actions.add_parser("index", help="index the library (library/library.toml)")
    index.add_argument("--fresh", action="store_true", help="re-index indexed sources")

    query = actions.add_parser("query", help="ask a question")
    query.add_argument("question")
    query.add_argument("--agent", choices=["graph", "agent"], default="graph")
    query.add_argument("--choice", type=int, help="pick reading N if the question is ambiguous")

    ans = actions.add_parser("eval-answers", help="graph vs agent on library/golden.yaml")
    ans.add_argument("--golden", type=Path, default=None, help="default: <library>/golden.yaml")
    ans.add_argument("--out", type=Path, default=Path("evals/ask"))


def print_answer(answer: Answer) -> None:
    """Print an answer with its references.

    Args:
        answer: The answer.
    """
    print(answer.text)
    for c in answer.citations:
        print(f"  [{c.n}] {c.label}  ({c.chunk_id})")
    if answer.dropped:
        print(f"  ({answer.dropped} unsupported sentence(s) removed)")


def main(config: Config, args: argparse.Namespace, ollama: httpx.BaseTransport | None) -> int:
    """Run an ask action.

    Args:
        config: Loaded configuration.
        args: Parsed arguments.
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code.
    """
    from labmate.ask.library import LibraryError, load_library  # noqa: PLC0415
    from labmate.ask.session import open_ask  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.library:
        config.ask.library = args.library
    s = open_ask(config, ollama)
    try:
        if args.action == "index":
            from labmate.ask.build import build_index  # noqa: PLC0415

            try:
                library = load_library(config.ask.library)
            except LibraryError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            done = build_index(s, library, fresh=args.fresh)
            print(f"Indexed {len(done)} source(s) into {config.ask.index_file}")
            return 0
        if not s.index.sources():
            print("error: the index is empty (run `make index`)", file=sys.stderr)
            return 1
        if args.action == "query":
            return _query(s, args)
        return _eval_answers(s, config, args)
    finally:
        s.close()


def _query(s: AskSession, args: argparse.Namespace) -> int:
    if args.agent == "agent":
        from labmate.ask.agent import ask_agent, build_agent  # noqa: PLC0415

        print_answer(ask_agent(s, build_agent(s), args.question))
        return 0
    from labmate.ask.graph import ask, compile_graph  # noqa: PLC0415

    def choose(question: str, options: list[str]) -> str:
        if args.choice is not None:
            return options[min(max(args.choice, 1), len(options)) - 1]
        if not sys.stdin.isatty():
            return options[0]
        print(f"'{question}' could mean:")  # pragma: no cover - interactive
        for i, option in enumerate(options, start=1):  # pragma: no cover
            print(f"  {i}. {option}")  # pragma: no cover
        picked = input("Which one? [1] ").strip() or "1"  # pragma: no cover
        return options[int(picked) - 1] if picked.isdigit() else options[0]  # pragma: no cover

    print_answer(ask(compile_graph(s), args.question, choose))
    return 0


def _eval_answers(s: AskSession, config: Config, args: argparse.Namespace) -> int:
    from labmate.ask.agent import ask_agent, build_agent  # noqa: PLC0415
    from labmate.ask.evals import answer_markdown, load_golden, score_answer  # noqa: PLC0415
    from labmate.ask.graph import ask, compile_graph  # noqa: PLC0415

    golden_path = args.golden or config.ask.library / "golden.yaml"
    if not golden_path.exists():
        print(f"error: no {golden_path} (a YAML list of questions)", file=sys.stderr)
        return 1
    graph, agent = compile_graph(s), build_agent(s)
    cases = []
    for g in load_golden(golden_path):
        for name in ("graph", "agent"):
            started = time.perf_counter()
            answer = ask(graph, g.question) if name == "graph" else ask_agent(s, agent, g.question)
            cases.append(score_answer(g, answer, started))
    args.out.mkdir(parents=True, exist_ok=True)
    table = answer_markdown(cases)
    (args.out / "answers.md").write_text(table)
    (args.out / "answers.json").write_text(
        json.dumps([c.model_dump() for c in cases], indent=2, ensure_ascii=False) + "\n"
    )
    print(table)
    return 0
