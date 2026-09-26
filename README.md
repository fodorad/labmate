# paper2carousel

> Turn an arXiv paper into a fact-checked LinkedIn carousel, fully local, fully reproducible.

**Status: pre-alpha (milestone M4a).** The agentic pipeline runs end to end: routing,
parallel claim extraction with a quote-verification guard, an orchestrated outline, a human
approval gate, grounded slide writing, a fact-check loop (a separate judge model plus exact
number checks, with rewrites) and a visuals agent that places paper figures or draws Graphviz
diagrams, all traced and replayable. Next: slide critic, alt texts and cover image (M4b),
then the evaluation suite (M5).

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
| render | plain code | Typst → 4:5 PDF |

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

Local PDFs work too: `uv run paper2carousel run --pdf path/to/paper.pdf --title "..."`.

`make probe` writes `probe/probe_report.md`.

## Replay modes

Set `[replay].mode` in `config.toml`:

| Mode | Behaviour |
|---|---|
| `live` | always call Ollama, store nothing |
| `record` | always call Ollama, (over)write cassettes |
| `auto` | cassette if present, otherwise call and record (dev default) |
| `replay` | cassettes only; a miss is an error (CI and published runs) |

## License

[AGPL-3.0-or-later](LICENSE). PyMuPDF, used for PDF parsing, is AGPL-licensed.
