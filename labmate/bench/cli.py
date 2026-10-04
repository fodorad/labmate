"""``labmate bench ...`` commands."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import httpx

from labmate.bench.cells import Cell, cells_markdown, load, save
from labmate.bench.guard import NotEnoughMemory, require_room
from labmate.bench.macro import FULL_USE_CASES, USE_CASES, run_cell
from labmate.bench.micro import (
    ModelResult,
    from_dict,
    measure_model,
    model_sizes,
    to_dict,
    unload_all,
)
from labmate.bench.report import HEADROOM_GB, machine, memory_gb, micro_markdown, profile_rows
from labmate.config import Config

log = logging.getLogger(__name__)


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate bench <action>``.

    Args:
        sub: The top-level subparsers.
    """
    bench = sub.add_parser("bench", help="measure models and use cases (not part of make check)")
    actions = bench.add_subparsers(dest="action", required=True)
    out = dict(type=Path, default=Path("evals/bench"), help="results folder")

    micro = actions.add_parser("micro", help="per model: load, speed, tools, structured, judging")
    micro.add_argument("--models", nargs="*", help="model tags (default: all in [profiles])")
    micro.add_argument("--out", **out)
    micro.add_argument("--force", action="store_true", help="measure models already measured")

    macro = actions.add_parser("macro", help="each use case on each model profile")
    macro.add_argument(
        "--profiles", nargs="*", help="default: those that fit in memory, and current"
    )
    macro.add_argument(
        "--use-cases", nargs="*", choices=FULL_USE_CASES, help="default: all but paper2*"
    )
    macro.add_argument("--full", action="store_true", help="also paper2flow and paper2post (slow)")
    macro.add_argument("--out", **out)
    macro.add_argument("--force", action="store_true", help="run cells already run")

    report = actions.add_parser("report", help="write docs/benchmarks.md from the saved results")
    report.add_argument("--out", **out)
    report.add_argument("--docs", type=Path, default=Path("docs/benchmarks.md"))


def profile_models(config: Config) -> tuple[list[str], list[str]]:
    """The writers and the judges named by the profiles (and ``[models]``).

    Args:
        config: Loaded configuration.

    Returns:
        ``(writers, judges)``, each without repeats.
    """
    profiles = [*config.profiles.values()]
    writers = [config.models.text, *(p.text for p in profiles)]
    judges = [config.models.critic, *(p.critic for p in profiles)]
    return list(dict.fromkeys(writers)), list(dict.fromkeys(judges))


PARALLEL = (
    "Two models loaded together both answer at once, but share the GPU (each at about half "
    "speed, 15% more in total). With `OLLAMA_NUM_PARALLEL=2`, two requests to `gemma4:26b-mlx` "
    "still run one after the other (the MLX runner ignores it), and two to `gemma4:e4b` run "
    "together at 10.3 tok/s each, against 19.7 alone: no gain in throughput, and each slot "
    "costs context memory. It stays at 1."
)
"""What was measured about parallel requests (a 200-token reply, same prompt)."""


def _micro_results(out: Path) -> list[ModelResult]:
    file = out / "micro.json"
    return [from_dict(d) for d in json.loads(file.read_text()).values()] if file.exists() else []


def run_micro(config: Config, args: argparse.Namespace) -> int:
    """Measure every model, writing the results as they come in.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``models``, ``out``, ``force``).

    Returns:
        Exit code 0.
    """
    writers, judges = profile_models(config)
    tags = args.models or list(dict.fromkeys([*writers, *judges]))
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    file = out / "micro.json"
    done: dict[str, dict] = {}
    if file.exists() and not args.force:
        done = json.loads(file.read_text())
    for tag in tags:
        if tag in done:
            log.info("  have %s", tag)
            continue
        unload_all(config.ollama.host)
        require_room(model_sizes(config.ollama.host).get(tag, 0.0), memory_gb())
        log.info("measuring %s", tag)
        try:
            done[tag] = to_dict(
                measure_model(config, tag, tools=tag in writers, judging=tag in judges)
            )
        finally:
            unload_all(config.ollama.host)
        file.write_text(json.dumps(done, indent=2) + "\n")
    text = micro_markdown(_micro_results(out), config, memory_gb())
    (out / "micro.md").write_text(f"# Model micro benchmark\n\n{machine()}\n\n{text}")
    print(text)
    return 0


def fitting_profiles(config: Config, out: Path) -> list[str]:
    """The profiles whose models fit in memory together.

    Args:
        config: Loaded configuration.
        out: Results folder holding ``micro.json``.

    Returns:
        Profile names.
    """
    sizes = {r.model: r.size_gb for r in _micro_results(out)}
    rows = profile_rows(config, sizes, memory_gb())
    return [r.profile for r in rows if r.fits]


def _profile_gb(config: Config, profile: str) -> float:
    """Size of the models a profile loads together."""
    chosen = config.profiles[profile]
    sizes = model_sizes(config.ollama.host)
    return sum(sizes.get(tag, 0.0) for tag in dict.fromkeys([chosen.text, chosen.critic]))


def run_macro(config: Config, args: argparse.Namespace) -> int:
    """Run each use case on each profile and save the cells.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``profiles``, ``use_cases``, ``full``, ``out``, ``force``).

    Returns:
        Exit code 0.
    """
    profiles = args.profiles or fitting_profiles(config, args.out)
    use_cases = args.use_cases or (FULL_USE_CASES if args.full else USE_CASES)
    directory = args.out / "cells"
    for use_case in use_cases:
        for profile in profiles:
            name = f"{use_case.replace(': ', '-')}__{profile}.json"
            if (directory / name).exists() and not args.force:
                log.info("  have %s on %s", use_case, profile)
                continue
            unload_all(config.ollama.host)
            require_room(_profile_gb(config, profile), memory_gb())
            log.info("running %s on %s", use_case, profile)
            try:
                cell = run_cell(use_case, profile, config)
            finally:
                unload_all(config.ollama.host)
            save(cell, directory)
            log.info("  %.0f s, quality %.0f%% %s", cell.seconds, cell.quality, cell.error)
    print(cells_markdown(load(directory)))
    return 0


def _ollama_version(host: str) -> str:
    try:
        return str(httpx.get(f"{host}/api/version", timeout=5).json().get("version", "unknown"))
    except (httpx.HTTPError, ValueError):
        return "unknown"


def run_report(config: Config, args: argparse.Namespace) -> int:
    """Write the benchmark page from the saved micro results and cells.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``out``, ``docs``).

    Returns:
        Exit code 0.
    """
    cells: list[Cell] = load(args.out / "cells") if (args.out / "cells").exists() else []
    intro = (
        f"Measured on {machine()}, Ollama {_ollama_version(config.ollama.host)}, {date.today()}. "
        "Every run starts cold (models unloaded) with the reply cache off, so it pays what a "
        "first request pays. The GPU may use about three quarters of the memory, 'fits' keeps "
        f"{HEADROOM_GB:.0f} GB of that free for other programs, and a pair that does not fit is "
        "evicted and loaded again during a run. Quality is the mean of deterministic checks the "
        "code already has (no model grades another); the parts are listed. Peak memory is read "
        "from `ollama ps`, which under-reports models run by llama-server (`gemma4:e4b`, "
        "`gemma4:e2b`). Run-to-run variation is about 30% (the same ask run took 25 and 32 s). "
        "`make bench-micro`, `make bench` and `make bench-report` reproduce this page."
    )
    page = [
        "# Models: speed, memory and quality",
        "",
        intro,
        "",
        "## Single models",
        "",
        micro_markdown(_micro_results(args.out), config, memory_gb()),
        "## Use cases",
        "",
        cells_markdown(cells) if cells else "No use case has been run yet.",
        "",
        "## Parallel requests",
        "",
        PARALLEL,
    ]
    args.docs.write_text("\n".join(page).rstrip("\n") + "\n")
    print(f"Wrote {args.docs}")
    return 0


def main(config: Config, args: argparse.Namespace) -> int:
    """Run a bench action.

    Args:
        config: Loaded configuration.
        args: Parsed arguments.

    Returns:
        Exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    actions = {"micro": run_micro, "macro": run_macro, "report": run_report}
    try:
        return actions[args.action](config, args)
    except NotEnoughMemory as e:
        log.error("stopped before loading anything more: %s", e)
        return 1
