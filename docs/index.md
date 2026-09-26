<!--
Keep near-identical to README.md (minus GitHub chrome). Update both in the same PR.
-->

# paper2carousel

> Turn an arXiv paper into a fact-checked LinkedIn carousel, fully local, fully reproducible.

**Status: pre-alpha (milestone M6).** The full pipeline runs: routing, parallel claim
extraction with a quote-verification guard, an orchestrated outline, a human approval gate,
grounded slide writing, a fact-check loop, a visuals agent (paper figures or Graphviz
diagrams), a generated cover image and a vision-model slide critic. All traced and
replayable, with an evaluation suite (run metrics and judge agreement against human
labels), an HTML trace viewer and a static gallery whose entries anyone can replay from
cassettes. Next: the first published papers and v0.1.0.

## What it will do

```
arXiv id ─▶ ingest ─▶ route ─▶ extract claims ─▶ plan outline ─▶ ✋ human approval
        ─▶ write slides ─▶ fact-check loop ─▶ visuals agent ─▶ render ─▶ carousel.pdf
```

- **Grounded:** every bullet on a slide cites a claim card, and every claim card carries
  a verbatim quote from the paper. A fact-check loop rewrites or drops unsupported bullets.
- **Agentic where it pays off:** routing, parallel extraction, an orchestrator, an
  evaluator–optimizer loop and one bounded tool-using agent. Everything else stays plain code.
- **Local and free:** runs on a Mac mini M4 (32 GB) with Ollama. No paid APIs.
- **Reproducible:** every model call is recorded to a cassette. `replay` mode reruns a
  published run byte-for-byte without any model installed; this is also what CI runs.

## How it works

| Step | Pattern | What it does |
|---|---|---|
| ingest | plain code | arXiv PDF → sections via the PDF outline, references dropped |
| route | routing | title + abstract → method / benchmark / survey / position template |
| extract | parallelisation | per section: claim cards with verbatim evidence quotes; quotes that aren't in the paper, or whose numbers differ, are dropped in code |
| outline | orchestrator | picks the slides and assigns claim cards; rules (template order, valid ids, ≤ 2 uses per claim) are checked in code and fed back on violation |
| gate | human-in-the-loop | writes `outline.yaml` and pauses; your edits are validated with the same rules |
| write | prompt chaining | one call per slide; every bullet cites the claim ids it uses |
| fact-check | evaluator–optimizer | numbers must match the cited evidence exactly; a different model judges each bullet against its evidence; failures go back to the writer with the reasons (≤ 2 rounds), then unsupported bullets are dropped |
| visuals | agent (tool use) | per slide the model calls `use_paper_figure`, `make_diagram` (Graphviz) or `no_visual`; tool errors come back as observations, ≤ 3 calls per slide; figures are cropped from the PDF during ingest |
| post | chaining + evaluator | LinkedIn post drafted from the final slides; its takeaways go through the same fact-check |
| cover | plain call | illustration from the local image model in the slide palette; skipped (not fatal) if generation fails |
| render | plain code | Typst → 4:5 PDF in the adamfodor.com palette, Inter font bundled for identical renders everywhere |
| critic | vision model | reviews each rendered page at phone size, writes alt texts, removes illegible or off-topic visuals |

## Models

| Role | Model |
|---|---|
| Writer / planner | `qwen3.6:35b-mlx` |
| Critic, fact-checker | `gemma4:26b-mlx` |
| Vision (slide critic, alt texts) | `gemma4:e4b` |
| Cover image | `x/z-image-turbo` or `x/flux2-klein` (decided by the probe) |

Only one large model fits in memory at a time, so the pipeline runs in phases
(text → critic → image) and unloads models between them.

**Probe findings** (`make probe`, Ollama 0.24, Mac mini M4): the MLX builds ignore Ollama's
`format=` JSON constraint (0/10 valid) but follow a schema given in the system prompt (10/10),
so the pipeline always sends both and validates with a retry loop. The MLX builds also ignore
image inputs, hence the separate non-MLX vision model.

## Quickstart

```bash
make install     # uv sync + pre-commit hooks
make check       # lint + type-check + tests + docs (no Ollama needed)

# with Ollama running:
make lock        # pin installed model digests into models.lock
make probe       # verify structured output, tools, vision, determinism; benchmark image models
make run ARXIV=1706.03762      # route, extract, plan -> pauses with runs/1706.03762/outline.yaml
make approve ARXIV=1706.03762  # after reviewing/editing the outline: write + render carousel.pdf
make replay ARXIV=1706.03762   # the whole run again from cassettes only, no Ollama needed
make baseline ARXIV=1706.03762 # the M1 one-shot version, for comparison
```

Each run folder contains `carousel.pdf`, `post.md` (paste-ready LinkedIn text),
`summary.md` (every bullet with its page and quote), `alt_texts.json`, `pages/*.png`, every
step's JSON artifact and `trace.jsonl`.

Local PDFs work too: `uv run paper2carousel run --pdf path/to/paper.pdf --title "..."`.

`make probe` writes `probe/probe_report.md`.

## Evaluation

```bash
make eval    # metrics of every finished run -> evals/results.md, results.json
make labels  # blind labelling sheet -> evals/labels.csv
make judges  # re-judge your labels with each judge model -> evals/judges.md
```

- **Run metrics** come from the run artifacts and the trace, no model needed: verified vs
  rejected claims, the share of first-draft bullets that failed the fact-check, bullets
  dropped after the loop, coverage of the paper's contribution claims, model calls,
  tokens and compute time (the paused and the approved invocation together).
- **Judge agreement:** `make labels` samples bullets from every fact-check round, about half
  of them rejected by the pipeline's judge, and writes them with their evidence but
  *without* the judge's verdict. You fill the `human` column (`s` / `p` / `u`).
  `make judges` then re-judges them with each candidate model using the pipeline's own
  judge prompt and reports accuracy and Cohen's κ, on the three labels and on pass/fail.
  The default candidates are the critic (`gemma4:26b-mlx`) and the writer judging itself
  (`qwen3.6:35b-mlx`), which tests whether a separate judge model is worth the swap.
  Judge calls are recorded to cassettes like everything else, so `--mode replay`
  reproduces the table.

## Trace viewer and gallery

```bash
make trace ARXIV=1706.03762    # runs/1706.03762/trace.html: every step and model call on a timeline
make publish ARXIV=1706.03762  # copy the finished run + the cassettes of its model calls to gallery/
make verify                    # replay every gallery entry from cassettes only, compare byte for byte
make site                      # static gallery -> site/ (GitHub Pages builds it on push to main)
```

A gallery entry contains the step artifacts, the carousel, the post, the trace and the
cassettes of exactly the model calls in that trace (not the paper, which is fetched from
arXiv again). CI runs `make verify`, so a published carousel that no longer reproduces
fails the build.

## Replay modes

Set `[replay].mode` in `config.toml`:

| Mode | Behaviour |
|---|---|
| `live` | always call Ollama, store nothing |
| `record` | always call Ollama, (over)write cassettes |
| `auto` | cassette if present, otherwise call and record (dev default) |
| `replay` | cassettes only; a miss is an error (CI and published runs) |

## License

[AGPL-3.0-or-later](https://github.com/fodorad/paper2carousel/blob/main/LICENSE). PyMuPDF, used for PDF parsing, is AGPL-licensed.

```{toctree}
:maxdepth: 2
:caption: Contents

api
```
