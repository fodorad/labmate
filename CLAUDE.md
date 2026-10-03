# labmate: notes for Claude Code

Agentic-AI features on one shared core. Everything runs on local
models (Ollama on a Mac mini M4, 32 GB), and every model call is recorded and replayable.

## Layout

- `labmate/core/`: shared by every feature.
  - `llm/`: Ollama client, record/replay cassettes (model calls and web requests), structured output with retries.
  - `model.py`: `LLM` role settings and the prompt loader.
  - `lc.py`: LangChain building blocks: `RecordedChatModel`, `structured` (validated replies), `step` (named chain steps), `batch_map`.
  - `callbacks.py`: `TraceHandler`, the one callback that writes chains, steps and model calls to `trace.jsonl`.
  - `tracing.py`: spans and `trace.jsonl`; `traceview.py` renders the HTML viewer.
  - `ingest.py`, `extract.py`: PDF sections and claim cards with verbatim quotes.
  - `factcheck.py`: the judge ⇄ rewrite loop.
  - `phases.py`: keeps one large model loaded at a time.
- `labmate/paper2flow/`: a **LangChain chain**, paper → `overview.pdf`.
  - `chain.py` builds `analyze` (ingest … flows) and `paper2flow = analyze | render`; `steps/` are the step functions.
  - Diagrams: the model proposes typed graphs, code checks them and writes Mermaid, mermaid-cli renders PNGs.
  - `evals/` holds the run metrics.
- `labmate/paper2post/`: a **LangChain chain**, paper → `post.pdf` (`analyze | post | icons | render`).
- `labmate/ask/`: the **agent** feature, agentic RAG over the library.
  - Library and index: `library.py` (tiers), `chunk.py` (outline sections, chunks, thesis points), `embed.py`, `index.py` (SQLite FTS5 + vectors, RRF), `build.py`.
  - LangGraph agent: `graph.py` (routing, clarify `interrupt`, `Send` fan-out to the research subgraph, conflict check, answer), `verify.py` (fact-check subgraph).
  - Baseline: `agent.py` is the prebuilt LangChain `create_agent`; `lc.py` holds the LangChain adapters over the recorded backend.
  - Evaluation: `evals.py` (retrieval recall@k/MRR from claim cards; answers on `library/golden.yaml`).
  - UI: `live.py` + `dashboard.py` (NiceGUI), `studio.py` (LangGraph Studio).
- `labmate/diagrams.py`: Mermaid diagrams. `make graphs` writes `docs/graphs.md`.
- `labmate/cli.py`: `labmate paper2flow|paper2post <paper>`, `ask …`, `eval`, `probe|lock|trace|graphs`. The `Makefile` wraps it.

## Commands

```bash
make install        # uv sync --extra dev + pre-commit hooks
make check          # ruff + mypy + pytest (>=90% coverage, currently ~99%) + sphinx -W
make flow PAPER=…   # paper2flow; make post PAPER=… for paper2post; MODE=replay reruns offline
make index          # ask: index library/ (library.toml + PDFs)
make ask Q="…"      # [THREAD=… AGENT=prebuilt]
make eval-rag       # evals/ask/retrieval.md
make eval-answers   # evals/ask/answers.md
make dashboard      # http://localhost:8080 ; make demo replays without Ollama
```

Tests never need Ollama: `tests/conftest.py` has a fake Ollama server (chat, tools,
embeddings), and `tests/ask/conftest.py` a fake model that plays every ask role.

## Conventions

- Google-style docstrings everywhere, including module constants (Sphinx autodoc).
- Line length 100, ruff + ruff-format, mypy on `labmate/` (pydantic plugin, `warn_unused_ignores`).
- TDD with ≥90% coverage. `labmate/ask/dashboard.py` is omitted from coverage.
- Conventional Commits (the history is the changelog). **No AI co-author
  trailers or AI attribution in commits**; a commit-msg hook rejects them.
- GitHub Flow: short-lived `feat/*`/`fix/*` branch → PR into `main` → green CI → merge.
  The gate is `make check` run locally; remote CI (`.github/workflows/ci.yml`) is manual-only for now. No CD, since labmate runs locally.
- Prompts are Markdown files next to the code (`*/prompts/*.md`), loaded with `load_prompt`.
- Model roles live in `config.toml` and digests in `models.lock` (`make lock`). A new
  model digest invalidates its cassettes.

## Replay and cassettes

`[replay].mode`:
- `auto` (dev default) uses a cassette when one exists, otherwise calls Ollama and records.
- `replay` never calls a model or the network.

The cassette key is sha256(model digest + canonical request), so changing a prompt,
schema or model re-records only the calls that changed. Long jobs (`make index`) can be
stopped and rerun: finished calls replay instantly.

## Gotchas

- `LABMATE_OLLAMA_HOST` overrides `[ollama].host`, e.g. `http://192.168.0.102:11434`
  when running inside a VM. Tests unset it.
- The MLX model builds ignore Ollama's `format=` JSON constraint, so structured calls also
  put the schema in the system prompt and validate with a retry loop (`core/lc.py`
  `structured`; ask still uses `core/llm/structured.py` directly).
- `RunnableLambda` passes the run config only to a parameter literally named `config`;
  `step` and `batch_map` wrap functions so their callbacks reach nested model calls.
- Diagrams need Node.js: mermaid-cli (pinned in `paper2flow/steps/flow.py`) runs headless
  Chromium; `make install` downloads it once.
- Parallel LangGraph branches share one SQLite connection; `Index` methods hold an RLock.
- `docs/conf.py` has to import LangChain up front and blank the docstrings of imported
  third-party names, otherwise autodoc fails on pydantic schema generation or invalid
  reStructuredText.
- LangGraph `add_node` typing differs between versions: see the `type: ignore` codes in
  `ask/graph.py` and `ask/verify.py`.
- `library/` PDFs and `*.sqlite` are gitignored; `library.toml` and `golden.yaml` are committed.
- Dissertation PDFs: LaTeX thesis classes letter-space headings ("I N T R O"), group
  chapters into unnumbered parts, and name the bibliography in the contents.
  `ask/chunk.py` handles all three; check `outline_sections` output on a new PDF first.
