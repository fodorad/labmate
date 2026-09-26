# paper2carousel

> Turn an arXiv paper into a fact-checked LinkedIn carousel, fully local, fully reproducible.

**Status: pre-alpha (milestone M1).** A walking skeleton runs end to end: arXiv id → PDF →
sections → one structured LLM call → rendered carousel, with tracing and record/replay. The
agentic steps (routing, claim extraction, outline, fact-check loop, visuals agent) come next.

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
make run ARXIV=1706.03762   # -> runs/1706.03762/carousel.pdf + trace.jsonl
make replay ARXIV=1706.03762  # same run from cassettes only, no Ollama needed
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
