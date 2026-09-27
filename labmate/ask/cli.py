"""``labmate ask ...`` commands (imported only when used: they need the ``[ask]`` extra)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from labmate.config import Config, ReplayMode
from labmate.core.llm.client import OllamaClient
from labmate.core.tracing import read_trace

if TYPE_CHECKING:
    from labmate.ask.schemas import Answer
    from labmate.ask.session import AskSession


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate ask <action>``.

    Args:
        sub: The top-level subparsers.
    """
    ask = sub.add_parser("ask", help="questions about your research, answered with citations")
    actions = ask.add_subparsers(dest="action", required=True)
    modes = [m.value for m in ReplayMode]

    index = actions.add_parser("index", help="index the library (library/library.toml)")
    index.add_argument("--fresh", action="store_true", help="re-index indexed sources")
    index.add_argument("--no-claims", action="store_true", help="skip claim extraction")
    index.add_argument("--mode", choices=modes, help="override [replay].mode")

    query = actions.add_parser("query", help="ask a question")
    query.add_argument("question")
    query.add_argument("--thread", default="default", help="conversation id (memory)")
    query.add_argument("--agent", choices=["graph", "prebuilt"], default="graph")
    query.add_argument("--choice", type=int, help="pick reading N if the question is ambiguous")
    query.add_argument("--mode", choices=modes, help="override [replay].mode")

    ret = actions.add_parser("eval-retrieval", help="recall@k / MRR -> evals/ask/retrieval.md")
    ret.add_argument("-n", type=int, default=60, help="questions to generate")
    ret.add_argument("--no-rewrite", action="store_true", help="skip hybrid + rewrite")
    ret.add_argument("--out", type=Path, default=Path("evals/ask"))
    ret.add_argument("--mode", choices=modes, help="override [replay].mode")

    ans = actions.add_parser("eval-answers", help="graph vs prebuilt on library/golden.yaml")
    ans.add_argument("--golden", type=Path, default=None, help="default: <library>/golden.yaml")
    ans.add_argument("--out", type=Path, default=Path("evals/ask"))
    ans.add_argument("--mode", choices=modes, help="override [replay].mode")

    dash = actions.add_parser("dashboard", help="live web UI of the agent")
    dash.add_argument("--port", type=int, default=8080)
    dash.add_argument("--demo", action="store_true", help="replay mode, paced (no Ollama needed)")
    dash.add_argument("--pace", type=float, default=0.6, help="seconds per step in demo mode")


def print_answer(answer: Answer) -> None:
    """Print an answer with its references.

    Args:
        answer: The answer.
    """
    print(answer.text)
    for c in answer.citations:
        print(f"  [{c.n}] {c.label}  ({c.chunk_id})")
    for x in answer.conflicts:
        print(f"  note: {x.topic}: dissertation {x.dissertation_value}, other {x.other_value}")
    if answer.dropped:
        print(f"  ({answer.dropped} unsupported sentence(s) removed by the fact-check)")


def main(config: Config, args: argparse.Namespace, client: OllamaClient | None) -> int:
    """Run an ask action.

    Args:
        config: Loaded configuration.
        args: Parsed arguments.
        client: Ollama client (tests inject a fake).

    Returns:
        Exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.action == "dashboard":
        from labmate.ask.dashboard import serve  # noqa: PLC0415 - starts a web server

        serve(config, args.port, demo=args.demo, pace=args.pace, client=client)
        return 0
    from labmate.ask.library import LibraryError, load_library  # noqa: PLC0415
    from labmate.ask.session import open_ask  # noqa: PLC0415

    mode = ReplayMode(args.mode) if getattr(args, "mode", None) else None
    s = open_ask(config, mode, client, label=f"ask-{args.action}")
    try:
        if args.action == "index":
            from labmate.ask.build import build_index  # noqa: PLC0415

            try:
                library = load_library(config.ask.library)
            except LibraryError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            done = build_index(s, library, fresh=args.fresh, claims=not args.no_claims)
            print(f"Indexed {len(done)} source(s) into {config.ask.index}")
            return 0
        if not s.index.sources():
            print("error: the index is empty (run `make index`)", file=sys.stderr)
            return 1
        if args.action == "query":
            return _query(s, args)
        if args.action == "eval-retrieval":
            from labmate.ask.evals import (  # noqa: PLC0415
                evaluate_retrieval,
                make_cases,
                save_retrieval,
            )

            cases = make_cases(s, args.n)
            scores = evaluate_retrieval(s, cases, rewrite=not args.no_rewrite)
            save_retrieval(args.out, cases, scores)
            print((args.out / "retrieval.md").read_text())
            return 0
        return _eval_answers(s, config, args)
    finally:
        s.close()


def _query(s: AskSession, args: argparse.Namespace) -> int:
    if args.agent == "prebuilt":
        from labmate.ask.agent import ask_prebuilt, build_agent  # noqa: PLC0415

        print_answer(ask_prebuilt(s, build_agent(s), args.question))
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

    print_answer(ask(s, compile_graph(s), args.question, args.thread, choose))
    return 0


def _eval_answers(s: AskSession, config: Config, args: argparse.Namespace) -> int:
    from labmate.ask.agent import ask_prebuilt, build_agent  # noqa: PLC0415
    from labmate.ask.evals import answer_markdown, load_golden, score_answer  # noqa: PLC0415
    from labmate.ask.graph import ask, compile_graph  # noqa: PLC0415

    golden_path = args.golden or config.ask.library / "golden.yaml"
    if not golden_path.exists():
        print(f"error: no {golden_path} (a YAML list of questions)", file=sys.stderr)
        return 1
    golden = load_golden(golden_path)
    graph, agent = compile_graph(s, args.out / "eval_threads.sqlite"), build_agent(s)
    cases = []
    for i, g in enumerate(golden):
        for name in ("graph", "prebuilt"):
            before = _spans(s)
            if name == "graph":
                answer = ask(s, graph, g.question, thread=f"eval-{i}")
            else:
                answer = ask_prebuilt(s, agent, g.question)
            spans = read_trace(s.tracer.path)[before:] if s.tracer.path else []
            cases.append(score_answer(s, g, answer, spans))
    args.out.mkdir(parents=True, exist_ok=True)
    table = answer_markdown(cases)
    (args.out / "answers.md").write_text(table)
    (args.out / "answers.json").write_text(
        json.dumps([c.model_dump() for c in cases], indent=2, ensure_ascii=False) + "\n"
    )
    print(table)
    return 0


def _spans(s: AskSession) -> int:
    """Spans in the session's trace file so far."""
    path = s.tracer.path
    return len(read_trace(path)) if path is not None and path.exists() else 0
