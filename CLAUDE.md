# labmate: notes for Claude Code

Agentic-AI features on one shared core. Everything runs on local models (Ollama on a Mac mini
M4, 32 GB). Each feature is a workflow (code decides the next step) or an agent (the model does).

## Layout

- `labmate/core/`: shared by every feature.
  - `chat.py`: `chat_model` (`ChatOllama` with the configured sampling), `embedder`, and the
    SQLite reply cache (`SQLiteCache`; `OllamaChat` puts the whole request in the cache key).
  - `lc.py`: LangChain building blocks: `prompt`, `structured` (validated replies with retries),
    `step` (named chain steps), `batch_map`. `structured.py` has the JSON parsing helpers.
  - `ingest.py`, `extract.py`: PDF sections and claim cards with verbatim quotes.
  - `factcheck.py`: the judge ⇄ rewrite loop.
- `labmate/paper2flow/`: a **LangChain chain**, paper → `overview.pdf`.
  - `chain.py` builds `analyze` (ingest, extract, write, factcheck, flows) and `paper2flow = analyze | render`; `steps/` are the step functions.
  - Diagrams: the model proposes typed graphs, code checks them and writes Mermaid, mermaid-cli renders PNGs.
  - `evals/metrics.py` computes run metrics from the artifacts.
- `labmate/paper2post/`: a **LangChain chain**, paper → `post.pdf` (`analyze | post | icons | render`).
- `labmate/ask/`: questions about a library of PDFs, with citations.
  - Library and index: `library.py`, `chunk.py` (outline sections, chunks, captions), `embed.py`, `index.py` (SQLite FTS5 + vectors, RRF), `build.py`.
  - `graph.py`: the **LangGraph graph** (understand, clarify `interrupt`, retrieve ⇄ grade ⇄ rewrite, answer, verify, abstain).
  - `agent.py`: the **tool-calling agent** (`create_agent`, three library tools).
  - `answer.py` (sentence checks, citations), `verify.py` (judge checks each sentence), `evidence.py`.
  - `evals.py`: graph vs agent on `library/golden.yaml`. `studio.py`: factories for LangGraph Studio.
- `labmate/scout/`: the **agent** `scout`: topic → `notes.md` (tools: arXiv search, abstract, paper2flow overview, write_notes that only accepts cited papers it read).
- `labmate/cv2job/`: a **LangChain chain with one agent step**, CV + posting → `cv.pdf`, `cover_letter.pdf`, `gap_report.pdf`.
  - `steps.py` (requirements, match, tailor, highlights; the checks), `gaps.py` (the agent), `letter.py` (fixed template), `render.py`, `chain.py`.
  - Input: `cv.yaml` (see `examples/cv2job/`) and a text posting. Your own files go in `private/` (git-ignored).
- `labmate/diagrams.py`: Mermaid diagrams drawn by LangChain/LangGraph. `make graphs` writes `docs/graphs.md`.
- `labmate/cli.py`: `labmate paper2flow|paper2post <paper>`, `scout <topic>`, `cv2job <cv> <job>`, `ask index|query|eval-answers`, `eval`, `graphs`. The `Makefile` wraps it.

## Commands

```bash
make install        # uv sync --extra dev + pre-commit hooks
make check          # ruff + mypy + pytest (>=90% coverage, currently ~99%) + sphinx -W
make flow PAPER=…   # paper2flow; make post PAPER=… for paper2post
make index          # ask: index library/ (library.toml + PDFs)
make ask Q="…"      # [AGENT=agent]
make cv CV=… JOB=…  # cv2job
make scout TOPIC=…  # scout (needs the network for arXiv)
make eval           # paper2flow run metrics
make eval-answers   # evals/ask/answers.md
```

Tests never need Ollama: `tests/conftest.py` has a fake Ollama server (chat, tools,
embeddings) that `ChatOllama` reaches through an httpx transport, and `tests/ask/conftest.py`
a fake model that plays every ask role.

## Conventions

- Google-style docstrings everywhere, including module constants (Sphinx autodoc).
- Line length 100, ruff + ruff-format, mypy on `labmate/` (pydantic plugin, `warn_unused_ignores`).
- TDD with ≥90% coverage; `labmate/ask/studio.py` is omitted from coverage.
- Conventional Commits (the history is the changelog). **No AI co-author
  trailers or AI attribution in commits**; a commit-msg hook rejects them.
- GitHub Flow: short-lived `feat/*`/`fix/*` branch → PR into `main` → green CI → merge.
  The gate is `make check` run locally; remote CI (`.github/workflows/ci.yml`) is manual-only for now. No CD, since labmate runs locally.
- Prompts are Markdown files next to the code (`*/prompts/*.md`), loaded with `load_prompt`.
- Model roles live in `config.toml`.

## Reply cache

`[cache]` in `config.toml`: model replies are cached in `cache/replies.sqlite` (gitignored). The
key holds the messages, model, sampling options, output schema and tools, so a changed prompt,
schema or model calls the model again and an unchanged request does not. Long jobs can be
stopped and rerun. Delete the file to start fresh.

## Gotchas

- `LABMATE_OLLAMA_HOST` overrides `[ollama].host`, e.g. `http://192.168.0.102:11434`
  when running inside a VM. Tests unset it.
- `ChatOllama`'s stock cache key is only the class name and stop words; use `chat_model`
  (an `OllamaChat`), never a bare `ChatOllama`, or replies leak between models and schemas.
- The MLX model builds ignore Ollama's `format=` JSON constraint, so `structured` also puts the
  schema in the system prompt and validates with a retry loop.
- `RunnableLambda` passes the run config only to a parameter literally named `config`;
  `step` and `batch_map` wrap functions so their callbacks reach nested model calls.
- Diagrams need Node.js: mermaid-cli (pinned in `paper2flow/steps/flow.py`) runs headless
  Chromium; `make install` downloads it once.
- `Index` methods hold an RLock (one SQLite connection shared by threads).
- `docs/conf.py` has to import LangChain up front and blank the docstrings of imported
  third-party names, otherwise autodoc fails on pydantic schema generation or invalid
  reStructuredText.
- LangGraph `add_node` typing differs between versions: see the `type: ignore` codes in
  `ask/graph.py`.
- Tables flattened into number rows are not chunked (`chunk.MAX_NUMERIC_SHARE`): models read
  their numbers off the wrong column.
- `library/` holds the committed PDFs (`pdf/`), `library.toml` and `golden.yaml`; `index.sqlite` is gitignored.
- Dissertation PDFs: LaTeX thesis classes letter-space headings ("I N T R O"), group
  chapters into unnumbered parts, and name the bibliography in the contents.
  `ask/chunk.py` handles all three; check `outline_sections` output on a new PDF first.
