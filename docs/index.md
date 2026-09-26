<!--
Keep near-identical to README.md (minus GitHub chrome). Update both in the same PR.
-->

# paper2carousel

> Turn an arXiv paper into a fact-checked LinkedIn carousel, fully local, fully reproducible.

**Status: pre-alpha, feature-complete for v0.1.** The full pipeline runs: routing,
parallel claim extraction with a quote-verification guard, an orchestrated outline, a
human approval gate, grounded slide writing, a fact-check loop, a visuals agent (paper
figures or Graphviz diagrams), a generated cover image and a vision-model slide critic.
All traced and replayable, with an evaluation suite (run metrics and judge agreement
against human labels), an HTML trace viewer, a static gallery whose entries anyone can
replay from cassettes, and two interchangeable orchestration engines (plain Python and
LangGraph). Next: the first published papers and v0.1.0.

## What it does

```
arXiv id / PDF ─▶ ingest ─▶ route ─▶ extract claims ─▶ plan the 4 blocks ─▶ ✋ human approval
               ─▶ write ─▶ fact-check loop ─▶ summary.pdf            (main output)
                                           └─▶ post.md + post.png     (LinkedIn post)
                                           └─▶ carousel.pdf           (optional)
```

Every paper becomes the same four blocks, the structure of a research project page:
**Task**, **Challenges**, **Proposed method** and **Main results**.

- **`summary.pdf`**, the main output: a one-page project summary (title, authors, main
  figure, abstract and the four blocks as cards).
- **LinkedIn post**: `post.md` (hook, 3–5 sentences telling the paper's story, a question)
  and `post.png`, a 1080×1350 image of the proposed method as a pipeline graph. The model
  proposes the graph as typed nodes and edges; code checks every label against the
  evidence and draws it with Graphviz in a fixed house style.
- **`carousel.pdf`** (`[outputs] carousel = true`): a cover plus one slide per block, each
  with a paper figure, a generated diagram or a chart of the paper's numbers.

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
| outline | orchestrator | assigns claim cards to the four blocks (task, challenges, method, results) and titles them; rules (the four blocks in order, valid ids, ≤ 2 uses per claim) are checked in code and fed back on violation |
| gate | human-in-the-loop | writes `outline.yaml` and pauses; your edits are validated with the same rules |
| write | prompt chaining | one call per slide; every bullet cites the claim ids it uses |
| fact-check | evaluator–optimizer | numbers must match the cited evidence exactly; a different model judges each bullet against its evidence; failures go back to the writer with the reasons (≤ 2 rounds), then unsupported bullets are dropped |
| visuals | agent (tool use) | per slide the model calls `use_paper_figure` (only offered the figures whose caption matches this slide best), `make_chart` (a bar chart whose every value must appear in the slide's evidence quotes, checked in code), `make_diagram` (Graphviz) or `no_visual`; tool errors come back as observations, ≤ 4 calls per slide |
| post | chaining + evaluator | LinkedIn post drafted from the final slides (hook, 3–5 sentences telling the paper's story, a question); every sentence goes through the same fact-check |
| graph | structured output + checks | the proposed method as typed nodes and edges; labels must name something in the evidence (checked in code, fed back on violation); Graphviz layout in the orientation that fills the 4:5 post image best |
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
make install     # uv sync (incl. the LangGraph extra) + pre-commit hooks
make check       # lint + type-check + tests + docs (no Ollama needed)

# with Ollama running:
make lock        # pin installed model digests into models.lock
make probe       # verify structured output, tools, vision, determinism; benchmark image models
make run ARXIV=1706.03762      # route, extract, plan -> pauses with runs/1706.03762/outline.yaml
make approve ARXIV=1706.03762  # after reviewing/editing the outline: write + render carousel.pdf
make replay ARXIV=1706.03762   # the whole run again from cassettes only, no Ollama needed
make baseline ARXIV=1706.03762 # the M1 one-shot version, for comparison
```

Each run folder contains `summary.pdf` (the one-page project summary), `post.md`
(paste-ready LinkedIn text) and `post.png` (its image), `carousel.pdf` if enabled,
`summary.md` (every bullet with its page and quote), `alt_texts.json`, `pages/*.png`, every
step's JSON artifact and `trace.jsonl`.

Papers that aren't on arXiv work too, from a path or a URL (title and authors come from
the PDF metadata when present; the URL is linked on the slides):

```bash
make run PDF=https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf
make approve PDF=https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf
make run PDF=papers/mine.pdf TITLE="My paper"
```

`make probe` writes `probe/probe_report.md`.

## Evaluation

```bash
make eval    # metrics of every finished run -> evals/results.md, results.json
make labels  # blind labelling sheet -> evals/labels.csv
make judges  # re-judge your labels with each judge model -> evals/judges.md
```

- **Run metrics** come from the run artifacts and the trace, no model needed: verified vs
  rejected claims, the share of first-draft bullets that failed the fact-check, bullets
  dropped after the loop, *block fit* (the share of bullets citing a claim of their
  block's kind, e.g. a result under Main results), the size of the post's method graph,
  model calls, tokens and compute time (the paused and the approved invocation together).
- **Judge agreement:** `make labels` samples bullets from every fact-check round, about half
  of them rejected by the pipeline's judge, and writes them with their evidence but
  *without* the judge's verdict. You fill the `human` column (`s` / `p` / `u`).
  `make judges` then re-judges them with each candidate model using the pipeline's own
  judge prompt and reports accuracy and Cohen's κ, on the three labels and on pass/fail.
  The default candidates are the critic (`gemma4:26b-mlx`) and the writer judging itself
  (`qwen3.6:35b-mlx`), which tests whether a separate judge model is worth the swap.
  Judge calls are recorded to cassettes like everything else, so `--mode replay`
  reproduces the table.

## Same pipeline, two ways

The steps know nothing about orchestration. Two engines drive them:

```bash
make run ARXIV=1706.03762                   # plain Python (default)
make run ARXIV=1706.03762 ENGINE=langgraph  # LangGraph StateGraph
```

| | `engine=plain` | `engine=langgraph` |
|---|---|---|
| Code (without docstrings) | ~100 lines | ~290 lines |
| Parallel extraction / writing | thread pool (`parallel_map`) | `Send` fan-out + `operator.add` reducer |
| Fact-check loop | `while` loop | `judge ⇄ rewrite` cycle with a conditional edge |
| Human gate | pause, re-run with `--approve`, reuse file checkpoints | `interrupt()`, resume from a SQLite checkpoint |
| Resume after a crash | step artifacts on disk | graph checkpoint per super-step |

Both call the same stage functions (`engines/common.py`) and the same per-item step
functions, so they send identical model requests. **In replay mode they write
byte-identical artifacts**; a test runs a paper with a rewrite round through both
engines and compares every file, so CI fails if they drift apart.

What LangGraph gave for free: checkpointing of the whole state after every node, a clean
interrupt/resume at the gate, and a graph picture of the pipeline. What it cost: about
3× the orchestration code, state that must be serializable (the checkpointer's
deserialization is allow-listed to this package's models), fan-out results that arrive
in any order (they carry their index and are sorted before assembly), and control flow
that is harder to step through in a debugger than a `for` loop. For a pipeline this
linear, the plain engine is the one I'd maintain; LangGraph earns its keep once there
are real branches, long-running interrupts or several agents sharing state.

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
