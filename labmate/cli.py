"""Command-line interface: ``labmate <command>``; every command is also a Makefile target.

- ``paper2flow`` / ``paper2post``: a paper in, ``overview.pdf`` / ``post.pdf`` out.
- ``ask``: questions about your research (see :mod:`labmate.ask.cli`).
- ``eval`` / ``labels`` / ``judges``: evaluate the paper chains' fact-check.
- ``probe`` / ``lock`` / ``trace`` / ``graphs``: model checks, pins, traces, diagrams.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
from langchain_core.runnables import Runnable

from labmate.config import Config, ReplayMode, load_config
from labmate.core.callbacks import TraceHandler
from labmate.core.ingest import IngestError
from labmate.core.lc import RecordedChatModel, step
from labmate.core.llm.client import OllamaClient, OllamaError, normalize_tag
from labmate.core.llm.replay import (
    CassetteMissError,
    CassetteStore,
    ReplayClient,
    read_lock,
    write_lock,
)
from labmate.core.model import LLM
from labmate.core.phases import ModelSwitcher
from labmate.core.probe import run_probe
from labmate.core.traceview import write_trace_html
from labmate.core.tracing import Tracer
from labmate.paper2flow.chain import ARTIFACTS, NothingSupportedError, paper2flow, run_id_for
from labmate.paper2flow.evals.agreement import agreement, agreement_markdown, judge_labels
from labmate.paper2flow.evals.labels import LabelledBullet, export_labels, read_labels
from labmate.paper2flow.evals.metrics import results_markdown, run_metrics
from labmate.paper2flow.schemas import VerdictLabel
from labmate.paper2post.chain import paper2post


def configured_models(config: Config) -> list[str]:
    """All model tags referenced by the config, fully qualified and de-duplicated.

    Args:
        config: Loaded configuration.

    Returns:
        Model tags in role order.
    """
    m = config.models
    tags = [m.text, m.critic, m.embed]
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
    write_lock(lock_path, {**old, **new})  # keeps other pinned models (extra judges)
    for tag, digest in new.items():
        flag = "" if old.get(tag) in (None, digest) else "  (CHANGED: old cassettes won't match)"
        print(f"{tag:32s} {digest[:12]}{flag}")
    print(f"Wrote {lock_path}")
    return 0


def cmd_probe(config: Config, client: OllamaClient, out: Path) -> int:
    """Run the capability probe on all configured models.

    Args:
        config: Loaded configuration.
        client: Ollama client.
        out: Output directory for the report.

    Returns:
        Exit code: 0 if no check failed, 1 otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    print(
        "Probing with a short built-in test passage (no paper needed). Each model is "
        "cold-loaded once; a 20 GB model can take a minute before the first check finishes.",
        flush=True,
    )
    report = run_probe(client, client.version(), [config.models.text, config.models.critic], out)
    print(report.to_markdown())
    print(f"Report written to {out}/probe_report.md")
    return 1 if any(r.passed is False for r in report.results) else 0


def cmd_paper(
    config: Config,
    args: argparse.Namespace,
    client: OllamaClient,
    transport: httpx.BaseTransport | None,
) -> int:
    """Run paper2flow or paper2post for one paper.

    Args:
        config: Loaded configuration.
        args: Parsed arguments (``command``, ``paper``, ``title``, ``mode``).
        client: Ollama client.
        transport: Live web transport (tests); the network otherwise.

    Returns:
        Exit code: 0 on success, 1 if the paper can't be read or nothing survives the
        fact-check.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    chain = paper2flow if args.command == "paper2flow" else paper2post
    mode = ReplayMode(args.mode) if args.mode else None
    try:
        out = chain(config, args.paper, args.title, mode, client, transport)
    except (IngestError, NothingSupportedError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{'Output:':8}{out}")
    print(f"{'Trace:':8}{out.parent / 'trace.html'}")
    return 0


def resolve_run(config: Config, ref: str) -> Path:
    """A run directory from its path, or from the paper it was made from.

    Args:
        config: Loaded configuration.
        ref: ``runs/1706.03762``, or the paper as given to the chain (arXiv id or URL, PDF
            URL, PDF path).

    Returns:
        The run directory (may not exist).
    """
    path = Path(ref)
    if path.is_dir():
        return path
    try:
        return config.tracing.runs_dir / run_id_for(ref)
    except IngestError:
        return config.tracing.runs_dir / ref


def cmd_trace(config: Config, ref: str, all_traces: bool) -> int:
    """Write the HTML trace viewer for a run.

    Args:
        config: Loaded configuration.
        ref: Run directory or paper.
        all_traces: Include every invocation, not just the latest finished run.

    Returns:
        Exit code: 0 on success, 1 if the run has no trace.
    """
    run_dir = resolve_run(config, ref)
    try:
        out = write_trace_html(run_dir, all_traces)
    except FileNotFoundError:
        print(f"error: no trace.jsonl in {run_dir}", file=sys.stderr)
        return 1
    print(f"Trace viewer: {out}")
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


def cmd_labels(config: Config, run_dirs: Sequence[Path], out: Path, n: int, seed: int) -> int:
    """Write or top up the blind labelling sheet.

    Args:
        config: Loaded configuration.
        run_dirs: Runs to sample from (default: all finished runs).
        out: CSV path.
        n: Target number of rows.
        seed: Sampling seed.

    Returns:
        Exit code: 0 on success, 1 if there is no finished run.
    """
    runs = finished_runs(config, run_dirs)
    if not runs:
        print(f"error: no finished runs (need {ARTIFACTS['checked']})", file=sys.stderr)
        return 1
    added = export_labels(runs, out, n, seed)
    print(
        f"Added {added} bullet(s) to {out}. Fill the `human` column with "
        "supported / partial / unsupported (or s / p / u), then run `make judges`."
    )
    return 0


def cmd_judges(
    config: Config,
    labels_path: Path,
    models: Sequence[str],
    out: Path,
    mode: ReplayMode | None,
    client: OllamaClient | None,
) -> int:
    """Re-judge the labelled bullets with each model and report agreement.

    Args:
        config: Loaded configuration.
        labels_path: Labelling sheet.
        models: Judge models (default: critic and writer, i.e. cross- vs self-judging).
        out: Output directory.
        mode: Replay mode override.
        client: Live Ollama client (``None`` in replay mode).

    Returns:
        Exit code: 0 on success, 1 without labelled bullets.
    """
    rows = read_labels(labels_path) if labels_path.exists() else []
    labelled = [b for b in rows if b.human is not None]
    if not labelled:
        print(f"error: no labelled bullets in {labels_path} (run `make labels`)", file=sys.stderr)
        return 1
    mode = mode or config.replay.mode
    live = None if mode is ReplayMode.REPLAY else client
    out.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(out / "judges_trace.jsonl")
    digests = read_lock(config.replay.lock_file)
    backend = ReplayClient(live, CassetteStore(config.replay.dir), mode, digests)
    gen = config.generation
    switcher = ModelSwitcher(live)

    def judging(model: str) -> Runnable[list[LabelledBullet], list[VerdictLabel]]:
        llm = LLM(backend, model, gen.seed, gen.temperature, gen.num_ctx, digests.get(model))
        judge = RecordedChatModel(llm=llm, switcher=switcher)
        workers = config.pipeline.workers
        return step("judge", lambda items, cfg: judge_labels(items, judge, workers, cfg))

    results = []
    for model in list(dict.fromkeys(models or [config.models.critic, config.models.text])):
        predicted = judging(model).invoke(
            labelled, {"callbacks": [TraceHandler(tracer, {"judge": model})], "run_name": "judges"}
        )
        results.append(agreement([b.human for b in labelled if b.human], predicted, model))
    switcher.release()
    table = agreement_markdown(results)
    (out / "judges.md").write_text(table)
    (out / "judges.json").write_text(json.dumps([r.model_dump() for r in results], indent=2) + "\n")
    print(table)
    print(f"Wrote {out / 'judges.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser: one command per feature, plus the shared commands.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="labmate", description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    modes = [m.value for m in ReplayMode]

    for name, output in (("paper2flow", "overview.pdf"), ("paper2post", "post.pdf")):
        chain = sub.add_parser(name, help=f"a paper -> runs/<id>/{output}")
        chain.add_argument("paper", help="arXiv id or URL, PDF URL or PDF path")
        chain.add_argument("--title", help="title override for PDFs without a usable title")
        chain.add_argument("--mode", choices=modes, help="override [replay].mode")

    from labmate.ask.cli import add_parser as add_ask  # noqa: PLC0415 - light imports only

    add_ask(sub)

    eval_p = sub.add_parser("eval", help="metrics of the finished paper runs -> evals/results.md")
    eval_p.add_argument("runs", nargs="*", type=Path, help="run dirs (default: all)")
    eval_p.add_argument("--out", type=Path, default=Path("evals"), help="output directory")

    labels = sub.add_parser("labels", help="blind labelling sheet of fact-checked bullets")
    labels.add_argument("runs", nargs="*", type=Path, help="run dirs (default: all)")
    labels.add_argument("--out", type=Path, default=Path("evals/labels.csv"), help="CSV path")
    labels.add_argument("-n", type=int, default=50, help="target number of bullets")
    labels.add_argument("--seed", type=int, default=0, help="sampling seed")

    judges = sub.add_parser("judges", help="agreement of judge models with your labels")
    judges.add_argument("--labels", type=Path, default=Path("evals/labels.csv"))
    judges.add_argument("--models", nargs="*", default=[], help="default: critic + writer")
    judges.add_argument("--out", type=Path, default=Path("evals"), help="output directory")
    judges.add_argument("--mode", choices=modes, help="override [replay].mode")

    probe = sub.add_parser("probe", help="verify model capabilities -> probe/probe_report.md")
    probe.add_argument("--out", type=Path, default=Path("probe"), help="report directory")

    sub.add_parser("lock", help="pin model digests into models.lock")

    trace = sub.add_parser("trace", help="HTML trace viewer for a run -> runs/<id>/trace.html")
    trace.add_argument("ref", help="run dir or paper, e.g. 1706.03762")
    trace.add_argument("--all", action="store_true", help="every invocation, not just the last")

    graphs = sub.add_parser("graphs", help="Mermaid diagrams of the chains and graphs -> docs/")
    graphs.add_argument("--out", type=Path, default=Path("docs/graphs.md"))
    return parser


def _model_error(error: OllamaError | CassetteMissError) -> int:
    """Report a failed model call: Ollama unreachable, or no cassette in replay mode.

    Args:
        error: The error.

    Returns:
        Exit code 2.
    """
    print(f"error: {error}", file=sys.stderr)
    if isinstance(error, CassetteMissError):
        print(
            "hint: a prompt, schema or model changed since the run was recorded; "
            "run in auto mode (with Ollama) to record the missing calls",
            file=sys.stderr,
        )
    return 2


def main(
    argv: Sequence[str] | None = None,
    client: OllamaClient | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    """Entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).
        client: Injected Ollama client (tests); built from the config otherwise.
        transport: Injected web transport (tests); the network otherwise.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "ask":
        from labmate.ask.cli import main as ask_main  # noqa: PLC0415 - needs the [ask] extra

        try:
            return ask_main(config, args, client)
        except (OllamaError, CassetteMissError) as e:
            return _model_error(e)
    if args.command == "graphs":
        from labmate.diagrams import write_diagrams  # noqa: PLC0415 - needs the [ask] extra

        print(f"Wrote {write_diagrams(args.out)}")
        return 0
    if args.command == "trace":
        return cmd_trace(config, args.ref, args.all)
    if args.command == "eval":
        return cmd_eval(config, args.runs, args.out)
    if args.command == "labels":
        return cmd_labels(config, args.runs, args.out, args.n, args.seed)
    own_client = client is None
    client = client or OllamaClient(config.ollama.host, config.ollama.timeout_s)
    try:
        if args.command in ("paper2flow", "paper2post"):
            return cmd_paper(config, args, client, transport)
        if args.command == "judges":
            mode = ReplayMode(args.mode) if args.mode else None
            return cmd_judges(config, args.labels, args.models, args.out, mode, client)
        if args.command == "lock":
            return cmd_lock(config, client)
        return cmd_probe(config, client, args.out)
    except (OllamaError, CassetteMissError) as e:
        return _model_error(e)
    finally:
        if own_client:
            client.close()
