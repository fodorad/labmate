"""Command-line interface. Every command is also exposed as a Makefile target."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from paper2carousel.config import Config, ReplayMode, load_config
from paper2carousel.engines.plain import run
from paper2carousel.llm.client import OllamaClient, OllamaError, normalize_tag
from paper2carousel.llm.replay import read_lock, write_lock
from paper2carousel.probe import run_probe


def configured_models(config: Config) -> list[str]:
    """All model tags referenced by the config, fully qualified and de-duplicated.

    Args:
        config: Loaded configuration.

    Returns:
        Model tags in role order.
    """
    m = config.models
    tags = [m.text, m.critic, m.vision, m.image, *m.image_candidates]
    return list(dict.fromkeys(normalize_tag(t) for t in tags))


def cmd_lock(config: Config, client: OllamaClient) -> int:
    """Pin the digests of all configured models into the lock file.

    Args:
        config: Loaded configuration.
        client: Ollama client.

    Returns:
        Exit code: 0 on success, 1 if a configured model is not installed.
    """
    installed = client.list_models()
    wanted = configured_models(config)
    missing = [t for t in wanted if t not in installed]
    if missing:
        print(f"Not installed (run `ollama pull`): {', '.join(missing)}", file=sys.stderr)
        return 1
    lock_path = config.replay.lock_file
    old = read_lock(lock_path)
    new = {t: installed[t] for t in wanted}
    write_lock(lock_path, new)
    for tag, digest in new.items():
        flag = "" if old.get(tag) in (None, digest) else "  (CHANGED: old cassettes won't match)"
        print(f"{tag:32s} {digest[:12]}{flag}")
    print(f"Wrote {lock_path}")
    return 0


def cmd_probe(config: Config, client: OllamaClient, out: Path, skip_images: bool) -> int:
    """Run the capability probe on all configured models.

    Args:
        config: Loaded configuration.
        client: Ollama client.
        out: Output directory for the report.
        skip_images: Skip the (slow) image-model benchmark.

    Returns:
        Exit code: 0 if no check failed, 1 otherwise.
    """
    m = config.models
    chat_models = [(m.text, False), (m.critic, False), (m.vision, True)]
    image_models = [] if skip_images else list(m.image_candidates)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    print(
        "Probing with a short built-in test passage (no paper needed). Each model is "
        "cold-loaded once; a 20 GB model can take a minute before the first check finishes.",
        flush=True,
    )
    report = run_probe(client, client.version(), chat_models, image_models, out)
    print(report.to_markdown())
    print(f"Report written to {out}/probe_report.md")
    return 1 if any(r.passed is False for r in report.results) else 0


def cmd_run(
    config: Config, args: argparse.Namespace, client: OllamaClient, http: httpx.Client | None
) -> int:
    """Run the pipeline for one paper.

    Args:
        config: Loaded configuration.
        args: Parsed ``run`` arguments.
        client: Ollama client.
        http: Optional HTTP client for arXiv (tests).

    Returns:
        Exit code: 0 on success, 1 if no paper was given.
    """
    if args.ref is None and args.pdf is None:
        print("error: give an arXiv id/URL or --pdf PATH", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    mode = ReplayMode(args.mode) if args.mode else None
    result = run(
        config,
        ref=args.ref,
        pdf=args.pdf,
        title=args.title,
        mode=mode,
        fresh=args.fresh,
        client=client,
        http=http,
    )
    print(f"Carousel: {result.carousel}\nTrace:    {result.trace}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="paper2carousel", description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="verify model capabilities and benchmark models")
    probe.add_argument("--out", type=Path, default=Path("probe"), help="report directory")
    probe.add_argument("--skip-images", action="store_true", help="skip image models")

    sub.add_parser("lock", help="pin model digests into models.lock")

    run_p = sub.add_parser("run", help="turn a paper into a carousel")
    run_p.add_argument("ref", nargs="?", help="arXiv id or URL, e.g. 1706.03762")
    run_p.add_argument("--pdf", type=Path, help="local PDF instead of arXiv")
    run_p.add_argument("--title", help="title override for --pdf")
    run_p.add_argument(
        "--mode", choices=[m.value for m in ReplayMode], help="override [replay].mode"
    )
    run_p.add_argument("--fresh", action="store_true", help="recompute LLM steps")
    return parser


def main(
    argv: Sequence[str] | None = None,
    client: OllamaClient | None = None,
    http: httpx.Client | None = None,
) -> int:
    """Entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).
        client: Injected Ollama client (tests); built from the config otherwise.
        http: Injected HTTP client for arXiv (tests).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    client = client or OllamaClient(config.ollama.host, config.ollama.timeout_s)
    try:
        if args.command == "lock":
            return cmd_lock(config, client)
        if args.command == "run":
            return cmd_run(config, args, client, http)
        return cmd_probe(config, client, args.out, args.skip_images)
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
