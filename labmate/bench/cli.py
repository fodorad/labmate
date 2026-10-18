"""``labmate bench ...`` commands."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from labmate.bench.cells import FULL_USE_CASES, USE_CASES, cell_key, cells_markdown, from_results
from labmate.bench.cold import ColdRun, load_results
from labmate.bench.guard import NotEnoughMemory
from labmate.bench.micro import ModelResult, from_dict, measure_model, to_dict
from labmate.bench.report import HEADROOM_GB, machine, memory_gb, micro_markdown, profile_rows
from labmate.bench.triage import DeciderResult, evaluate, load_golden, triage_markdown
from labmate.bench.triage import to_dict as decider_dict
from labmate.config import Config
from labmate.core.ollama import OllamaServer

if TYPE_CHECKING:
    from labmate.cli import CliContext

log = logging.getLogger(__name__)

Transport = httpx.BaseTransport | None
"""A transport to Ollama (tests), or ``None`` for the configured host."""


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

    triage = actions.add_parser("triage", help="compare triage deciders on a labelled set")
    triage.add_argument("--models", nargs="*", help="default: the decision models and gemma4:e4b")
    triage.add_argument("--out", **out)
    triage.add_argument("--force", action="store_true", help="measure models already measured")

    report = actions.add_parser("report", help="write docs/benchmarks.md from the saved results")
    report.add_argument("--out", **out)
    report.add_argument("--docs", type=Path, default=Path("docs/benchmarks.md"))
    bench.set_defaults(handler=main)


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


TRIAGE_NOTE = (
    "36 cases: 18 arXiv papers, each labelled `deep`, `post` or `skip` by hand for two sets of "
    "interests (`evals/triage/golden.yaml`). A decision model scores relevance in one pass and "
    "code maps the score to an action (`>= 3.0` deep, `>= 1.75` post); a chat model is asked for "
    "the action as validated JSON. The two thresholds were picked on these cases; picked on one "
    "set of interests and tested on the other, `clef-flash` gets 30, `nimble` 29 and `tev1` 28 of "
    "36 right. `clef` (27B, 18 GB) does not fit next to the writer."
)
"""How the triage deciders were compared."""


PARALLEL = (
    "Two models loaded together both answer at once, but share the GPU (each at about half "
    "speed, 15% more in total). With `OLLAMA_NUM_PARALLEL=2`, two requests to `gemma4:26b-mlx` "
    "still run one after the other (the MLX runner ignores it), and two to `gemma4:e4b` run "
    "together at 10.3 tok/s each, against 19.7 alone: no gain in throughput, and each slot "
    "costs context memory. It stays at 1."
)
"""What was measured about parallel requests (a 200-token reply, same prompt)."""


def _micro_results(out: Path) -> list[ModelResult]:
    return [from_dict(d) for d in load_results(out / "micro.json").values()]


def run_micro(config: Config, args: argparse.Namespace, ollama: Transport) -> int:
    """Measure every model, writing the results as they come in.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``models``, ``out``, ``force``).
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code 0.
    """
    writers, judges = profile_models(config)
    tags = args.models or list(dict.fromkeys([*writers, *judges]))
    cold = ColdRun(config, args.out / "micro.json", args.force, ollama)
    sizes = cold.server.sizes()
    for tag in tags:
        size, tools, judging = sizes.get(tag, 0.0), tag in writers, tag in judges
        cold.run(
            tag,
            size,
            lambda cfg, usage: to_dict(measure_model(cfg, usage, tag, size, tools, judging)),  # noqa: B023 - called at once
        )
    text = micro_markdown(_micro_results(args.out), config, memory_gb())
    (args.out / "micro.md").write_text(f"# Model micro benchmark\n\n{machine()}\n\n{text}")
    print(text)
    return 0


DECIDERS = ("clef-flash", "nimble", "tev1", "tev1:0.8b", "gemma4:e4b")
"""The decision models on Ollama that fit this Mac, and a chat model as the baseline."""


def _deciders(out: Path) -> list[DeciderResult]:
    return [DeciderResult(**d) for d in load_results(out / "triage.json").values()]


def run_triage_eval(config: Config, args: argparse.Namespace, ollama: Transport) -> int:
    """Rate the labelled papers with each decider, saving as it goes.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``models``, ``out``, ``force``).
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code 0.
    """
    golden = load_golden()
    cold = ColdRun(config, args.out / "triage.json", args.force, ollama)
    sizes = cold.server.sizes()
    for tag in args.models or DECIDERS:
        cold.run(
            tag,
            sizes.get(tag, 0.0),
            lambda cfg, usage: decider_dict(evaluate(cfg, tag, golden, ollama)),  # noqa: B023 - called at once
        )
    print(triage_markdown(_deciders(args.out), sizes))
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


def run_macro(config: Config, args: argparse.Namespace, ollama: Transport) -> int:
    """Run each use case on each profile and save the cells.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``profiles``, ``use_cases``, ``full``, ``out``, ``force``).
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code 0.
    """
    from labmate.bench.macro import run_cell  # noqa: PLC0415 - runs every use case ([ask] extra)

    profiles = args.profiles or fitting_profiles(config, args.out)
    use_cases = args.use_cases or (FULL_USE_CASES if args.full else USE_CASES)
    cold = ColdRun(config, args.out / "macro.json", args.force, ollama)
    sizes = cold.server.sizes()
    for use_case in use_cases:
        for profile in profiles:
            chosen = config.profiles[profile]
            gb = sum(sizes.get(tag, 0.0) for tag in dict.fromkeys([chosen.text, chosen.critic]))
            cold.run(
                cell_key(use_case, profile),
                gb,
                lambda cfg, usage: asdict(run_cell(use_case, profile, cfg, usage)),  # noqa: B023 - called at once
            )
    print(cells_markdown(from_results(cold.results())))
    return 0


def run_report(config: Config, args: argparse.Namespace, ollama: Transport) -> int:
    """Write the benchmark page from the saved micro results and cells.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``out``, ``docs``).
        ollama: Transport to the Ollama server (tests); the configured host otherwise.

    Returns:
        Exit code 0.
    """
    server = OllamaServer.from_config(config, ollama)
    cells = from_results(load_results(args.out / "macro.json"))
    intro = (
        f"Measured on {machine()}, Ollama {server.version()}, {date.today()}. "
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
        "## Triage deciders",
        "",
        TRIAGE_NOTE,
        "",
        triage_markdown(_deciders(args.out), server.sizes())
        if _deciders(args.out)
        else "Not measured yet (`labmate bench triage`).",
        "## Parallel requests",
        "",
        PARALLEL,
    ]
    args.docs.write_text("\n".join(page).rstrip("\n") + "\n")
    print(f"Wrote {args.docs}")
    return 0


def main(config: Config, args: argparse.Namespace, ctx: CliContext) -> int:
    """Run a bench action.

    Args:
        config: Loaded configuration.
        args: Parsed arguments.
        ctx: The transport to Ollama.

    Returns:
        Exit code.
    """
    actions = {
        "micro": run_micro,
        "macro": run_macro,
        "triage": run_triage_eval,
        "report": run_report,
    }
    try:
        return actions[args.action](config, args, ctx.ollama)
    except NotEnoughMemory as e:
        log.error("stopped before loading anything more: %s", e)
        return 1
