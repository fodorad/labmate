"""``labmate bench ...`` commands."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from labmate.bench.micro import ModelResult, measure_model, to_dict
from labmate.bench.report import machine, memory_gb, micro_markdown
from labmate.config import Config

log = logging.getLogger(__name__)


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg,unused-ignore]
    """Register ``labmate bench <action>``.

    Args:
        sub: The top-level subparsers.
    """
    bench = sub.add_parser("bench", help="measure models and use cases (not part of make check)")
    actions = bench.add_subparsers(dest="action", required=True)
    micro = actions.add_parser("micro", help="per model: load, speed, tools, structured, judging")
    micro.add_argument("--models", nargs="*", help="model tags (default: all in [profiles])")
    micro.add_argument("--out", type=Path, default=Path("evals/bench"))
    micro.add_argument("--force", action="store_true", help="measure models already measured")


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
    results: list[ModelResult] = []
    for tag in tags:
        if tag in done:
            log.info("  have %s", tag)
            continue
        log.info("measuring %s", tag)
        result = measure_model(config, tag, tools=tag in writers, judging=tag in judges)
        done[tag] = to_dict(result)
        file.write_text(json.dumps(done, indent=2) + "\n")
        results.append(result)
    from_file = [_result(d) for d in done.values()]
    text = f"# Model micro benchmark\n\n{machine()}\n\n" + micro_markdown(
        from_file, config, memory_gb()
    )
    (out / "micro.md").write_text(text)
    print(text)
    return 0


def _result(data: dict) -> ModelResult:
    from labmate.bench.micro import JudgeResult  # noqa: PLC0415

    judge = data.pop("judge")
    return ModelResult(**data, judge=JudgeResult(**judge) if judge else None)


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
    return run_micro(config, args)
