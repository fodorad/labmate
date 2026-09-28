# labmate: notes for Claude Code

A portfolio repo of agentic-AI features on one shared core. Everything runs on local
models (Ollama on a Mac mini M4, 32 GB), and every model call is recorded and replayable.

## Layout

- `labmate/core/`: shared by every feature.
  - `llm/`: Ollama client, record/replay cassettes, structured output with retries.
  - `model.py`: `LLM` role settings and the prompt loader.
  - `tracing.py`: spans to `trace.jsonl`; `traceview.py` renders the HTML viewer.
  - `ingest.py`, `extract.py`: PDF sections and claim cards with verbatim quotes.
  - `factcheck.py`: the judge ⇄ rewrite loop.
  - `phases.py`: keeps one large model loaded at a time.
- `labmate/paper2flow/`: the **workflow** feature.
  - Paper → `overview.pdf` + `post.pdf`.
  - `steps/` are the steps; `engines/` runs them in plain Python and in LangGraph (both kept on purpose).
  - `evals/` holds the metrics, the labelling sheet and the judge agreement.
- `labmate/ask/`: the **agent** feature, agentic RAG over the library.
  - Library and index: `library.py` (tiers), `chunk.py` (outline sections, chunks, thesis points), `embed.py`, `index.py` (SQLite FTS5 + vectors, RRF), `build.py`.
  - LangGraph agent: `graph.py` (routing, clarify `interrupt`, `Send` fan-out to the research subgraph, conflict check, answer), `verify.py` (fact-check subgraph).
  - Baseline: `agent.py` is the prebuilt LangChain `create_agent`; `lc.py` holds the LangChain adapters over the recorded backend.
  - Evaluation: `evals.py` (retrieval recall@k/MRR from claim cards; answers on `library/golden.yaml`).
  - UI: `live.py` + `dashboard.py` (NiceGUI), `studio.py` (LangGraph Studio).
- `labmate/diagrams.py`: Mermaid diagrams. `make graphs` writes `docs/graphs.md`.
- `labmate/cli.py`: `labmate probe|lock|trace|graphs|paper2flow …|ask …`. The `Makefile` wraps it.

## Commands

```bash
make install        # uv sync --extra dev + pre-commit hooks
make check          # ruff + mypy + pytest (>=90% coverage, currently ~99%) + sphinx -W
make run ARXIV=…    # paper2flow; make approve ARXIV=… after editing outline.yaml
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
- Conventional Commits (release-please builds the changelog). **No AI co-author
  trailers or AI attribution in commits**; a commit-msg hook rejects them.
- Branches: feature → `dev` → `main`. Adam pushes and opens PRs himself.
- Prompts are Markdown files next to the code (`*/prompts/*.md`), loaded with `load_prompt`.
- Model roles live in `config.toml` and digests in `models.lock` (`make lock`). A new
  model digest invalidates its cassettes.

## Replay and cassettes

`[replay].mode`:
- `auto` (dev default) uses a cassette when one exists, otherwise calls Ollama and records.
- `replay` never calls a model (CI, gallery).

The cassette key is sha256(model digest + canonical request), so changing a prompt,
schema or model re-records only the calls that changed. Long jobs (`make index`) can be
stopped and rerun: finished calls replay instantly.

## Gotchas

- `LABMATE_OLLAMA_HOST` overrides `[ollama].host`, e.g. `http://192.168.0.102:11434`
  when running inside a VM. Tests unset it.
- The MLX model builds ignore Ollama's `format=` JSON constraint, so structured calls also
  put the schema in the system prompt and validate with a retry loop
  (`core/llm/structured.py`).
- Parallel LangGraph branches share one SQLite connection; `Index` methods hold an RLock.
- `docs/conf.py` has to import LangChain up front and blank the docstrings of imported
  third-party names, otherwise autodoc fails on pydantic schema generation or invalid
  reStructuredText.
- LangGraph `add_node` typing differs between versions: see the `type: ignore` codes in
  both engines.
- `library/` PDFs and `*.sqlite` are gitignored; `library.toml` and `golden.yaml` are committed.
- Dissertation PDFs: LaTeX thesis classes letter-space headings ("I N T R O"), group
  chapters into unnumbered parts, and name the bibliography in the contents.
  `ask/chunk.py` handles all three; check `outline_sections` output on a new PDF first.
