"""Command-line interface. Every command is also exposed as a Makefile target."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from paper2carousel.config import Config, ReplayMode, load_config
from paper2carousel.engines.plain import run
from paper2carousel.evals.agreement import agreement, agreement_markdown, judge_labels
from paper2carousel.evals.labels import export_labels, read_labels
from paper2carousel.evals.metrics import results_markdown, run_metrics
from paper2carousel.gallery import PublishError, build_site, publish, verify
from paper2carousel.llm.client import OllamaClient, OllamaError, normalize_tag
from paper2carousel.llm.replay import CassetteStore, ReplayClient, read_lock, write_lock
from paper2carousel.phases import ModelSwitcher
from paper2carousel.probe import run_probe
from paper2carousel.steps.ingest import (
    USER_AGENT,
    IngestError,
    download_url,
    is_url,
    parse_arxiv_id,
    slugify,
    url_stem,
)
from paper2carousel.steps.llm import LLM
from paper2carousel.traceview import write_trace_html
from paper2carousel.tracing import TracedClient, Tracer


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
    pdf, url = (Path(args.pdf) if args.pdf else None), args.url or ""
    if args.pdf and is_url(args.pdf):
        own = http is None
        client_http = http or httpx.Client(
            timeout=120, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        try:
            pdf = download_url(args.pdf, config.tracing.runs_dir / ".downloads", client_http)
        finally:
            if own:
                client_http.close()
        url = url or args.pdf
    common = {
        "ref": args.ref,
        "pdf": pdf,
        "title": args.title,
        "url": url,
        "mode": mode,
        "fresh": args.fresh,
        "approve": args.approve,
        "auto_approve": args.auto_approve,
        "client": client,
        "http": http,
    }
    if args.engine == "langgraph":
        if args.baseline:
            print("error: --baseline is only available with --engine plain", file=sys.stderr)
            return 1
        from paper2carousel.engines import langgraph_engine  # optional dependency

        result = langgraph_engine.run(config, **common)
    else:
        result = run(config, baseline=args.baseline, **common)
    if result.status == "awaiting_approval":
        print(
            f"Outline ready for review: {result.gate}\n"
            "Edit it if you like, then continue with --approve (make approve)."
        )
        return 0
    viewer = write_trace_html(result.run_dir)
    lines = [
        ("Summary", result.summary),
        ("Post", result.post_image),
        ("Carousel", result.carousel),
    ]
    for name, path in lines:
        if path is not None and path.exists():
            print(f"{name + ':':10}{path}")
    print(f"{'Trace:':10}{viewer}")
    return 0


def resolve_run(config: Config, ref: str) -> Path:
    """A run directory from a path or an arXiv id / slug under ``runs_dir``.

    Args:
        config: Loaded configuration.
        ref: ``runs/1706.03762``, ``1706.03762``, ``arXiv:1706.03762``, or the PDF path or
            URL the run was made from.

    Returns:
        The run directory (may not exist).
    """
    path = Path(ref)
    if path.is_dir():
        return path
    if is_url(ref) or ref.lower().endswith(".pdf"):
        return config.tracing.runs_dir / slugify(url_stem(ref) if is_url(ref) else path.stem)
    try:
        return config.tracing.runs_dir / parse_arxiv_id(ref)
    except IngestError:
        return config.tracing.runs_dir / slugify(ref)


def cmd_trace(config: Config, ref: str, all_traces: bool) -> int:
    """Write the HTML trace viewer for a run.

    Args:
        config: Loaded configuration.
        ref: Run directory or paper id.
        all_traces: Include every invocation, not just the latest completed run.

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


def cmd_publish(
    config: Config, ref: str, gallery: Path, include_pdf: bool, http: httpx.Client | None = None
) -> int:
    """Publish a finished run into the gallery.

    Args:
        config: Loaded configuration.
        ref: Run directory or paper id.
        gallery: Gallery root.
        include_pdf: Also publish the paper PDF.
        http: HTTP client for re-fetching the paper during the replay check (tests).

    Returns:
        Exit code: 0 on success, 1 if the run can't be published.
    """
    try:
        entry, n = publish(resolve_run(config, ref), gallery, config, include_pdf, http)
    except (PublishError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Published {entry}: replays exactly from its {n} cassette(s).")
    return 0


def cmd_verify(
    config: Config, gallery: Path, papers: Sequence[str], http: httpx.Client | None
) -> int:
    """Replay published gallery entries from their cassettes and compare the artifacts.

    Args:
        config: Loaded configuration.
        gallery: Gallery root.
        papers: Entry ids (default: all).
        http: HTTP client for arXiv (tests).

    Returns:
        Exit code: 0 if every entry reproduced exactly, 1 otherwise.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
    entries = [gallery / p for p in papers] or sorted(m.parent for m in gallery.glob("*/meta.json"))
    if not entries:
        print(f"Nothing to verify in {gallery}")
        return 0
    failed = 0
    for entry in entries:
        report = verify(entry, config, http)
        status = "OK  " if report.ok else "FAIL"
        print(f"{status} {report.paper_id}: {len(report.identical)} identical", end="")
        if report.different:
            print(f", different: {', '.join(report.different)}", end="")
        print(f" ({report.error})" if report.error else "")
        failed += not report.ok
    return 1 if failed else 0


def cmd_site(gallery: Path, out: Path, judges: Path) -> int:
    """Build the static gallery site.

    Args:
        gallery: Gallery root.
        out: Output directory.
        judges: ``judges.json`` to include judge agreement, if it exists.

    Returns:
        Exit code 0.
    """
    metrics = build_site(gallery, out, judges)
    print(f"Built {out}/index.html with {len(metrics)} paper(s).")
    return 0


def finished_runs(config: Config, run_dirs: Sequence[Path]) -> list[Path]:
    """The given run directories, or every run that reached the fact-check.

    Args:
        config: Loaded configuration (for ``runs_dir``).
        run_dirs: Explicit run directories; empty means "all".

    Returns:
        Run directories containing ``05_factcheck.json``, sorted.
    """
    candidates = list(run_dirs) or sorted(config.tracing.runs_dir.glob("*"))
    return [d for d in candidates if (d / "05_factcheck.json").exists()]


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
        print("error: no finished runs (need 05_factcheck.json)", file=sys.stderr)
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
        print("error: no finished runs (need 05_factcheck.json)", file=sys.stderr)
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
    backend = TracedClient(
        ReplayClient(live, CassetteStore(config.replay.dir), mode, digests), tracer, digests
    )
    gen = config.generation
    switcher = ModelSwitcher(live)
    results = []
    for model in list(dict.fromkeys(models or [config.models.critic, config.models.text])):
        switcher.use(model)
        judge = LLM(backend, model, gen.seed, gen.temperature, gen.num_ctx)
        predicted = judge_labels(labelled, judge, config.pipeline.workers)
        results.append(agreement([b.human for b in labelled if b.human], predicted, model))
    switcher.release()
    table = agreement_markdown(results)
    (out / "judges.md").write_text(table)
    (out / "judges.json").write_text(json.dumps([r.model_dump() for r in results], indent=2) + "\n")
    print(table)
    print(f"Wrote {out / 'judges.md'}")
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
    run_p.add_argument("--pdf", help="PDF path or URL instead of arXiv")
    run_p.add_argument("--url", help="link shown on the slides for --pdf (default: the URL)")
    run_p.add_argument("--title", help="title override for --pdf")
    run_p.add_argument(
        "--mode", choices=[m.value for m in ReplayMode], help="override [replay].mode"
    )
    run_p.add_argument("--fresh", action="store_true", help="recompute LLM steps")
    run_p.add_argument("--approve", action="store_true", help="accept outline.yaml, continue")
    run_p.add_argument("--auto-approve", action="store_true", help="skip the human gate")
    run_p.add_argument("--baseline", action="store_true", help="M1 one-shot pipeline")
    run_p.add_argument(
        "--engine",
        choices=["plain", "langgraph"],
        default="plain",
        help="orchestration engine (langgraph needs the [langgraph] extra)",
    )

    eval_p = sub.add_parser("eval", help="metrics of finished runs -> evals/results.md")
    eval_p.add_argument("runs", nargs="*", type=Path, help="run dirs (default: all)")
    eval_p.add_argument("--out", type=Path, default=Path("evals"), help="output directory")

    labels = sub.add_parser("labels", help="blind labelling sheet of fact-checked bullets")
    labels.add_argument("runs", nargs="*", type=Path, help="run dirs (default: all)")
    labels.add_argument("--out", type=Path, default=Path("evals/labels.csv"), help="CSV path")
    labels.add_argument("-n", type=int, default=50, help="target number of bullets")
    labels.add_argument("--seed", type=int, default=0, help="sampling seed")

    trace = sub.add_parser("trace", help="HTML trace viewer for a run -> runs/<id>/trace.html")
    trace.add_argument("ref", help="run dir or paper id, e.g. 1706.03762")
    trace.add_argument("--all", action="store_true", help="every invocation, not just the last")

    pub = sub.add_parser("publish", help="copy a finished run + its cassettes into the gallery")
    pub.add_argument("ref", help="run dir or paper id")
    pub.add_argument("--gallery", type=Path, default=Path("gallery"))
    pub.add_argument("--include-pdf", action="store_true", help="also publish the paper PDF")

    ver = sub.add_parser("verify", help="replay gallery entries from cassettes, compare outputs")
    ver.add_argument("papers", nargs="*", help="entry ids (default: all)")
    ver.add_argument("--gallery", type=Path, default=Path("gallery"))

    site = sub.add_parser("site", help="build the static gallery site")
    site.add_argument("--gallery", type=Path, default=Path("gallery"))
    site.add_argument("--out", type=Path, default=Path("site"))
    site.add_argument("--judges", type=Path, default=Path("evals/judges.json"))

    judges = sub.add_parser("judges", help="agreement of judge models with your labels")
    judges.add_argument("--labels", type=Path, default=Path("evals/labels.csv"))
    judges.add_argument("--models", nargs="*", default=[], help="default: critic + writer")
    judges.add_argument("--out", type=Path, default=Path("evals"), help="output directory")
    judges.add_argument(
        "--mode", choices=[m.value for m in ReplayMode], help="override [replay].mode"
    )
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
    if args.command == "eval":
        return cmd_eval(config, args.runs, args.out)
    if args.command == "labels":
        return cmd_labels(config, args.runs, args.out, args.n, args.seed)
    if args.command == "trace":
        return cmd_trace(config, args.ref, args.all)
    if args.command == "publish":
        return cmd_publish(config, args.ref, args.gallery, args.include_pdf, http)
    if args.command == "verify":
        return cmd_verify(config, args.gallery, args.papers, http)
    if args.command == "site":
        return cmd_site(args.gallery, args.out, args.judges)
    client = client or OllamaClient(config.ollama.host, config.ollama.timeout_s)
    try:
        if args.command == "judges":
            mode = ReplayMode(args.mode) if args.mode else None
            return cmd_judges(config, args.labels, args.models, args.out, mode, client)
        if args.command == "lock":
            return cmd_lock(config, client)
        if args.command == "run":
            return cmd_run(config, args, client, http)
        return cmd_probe(config, client, args.out, args.skip_images)
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
