# labmate

> Local tools on Ollama: paper overviews and posts, cited answers about a paper library, an arXiv scout, and CV tailoring.

All features run on local models through [Ollama](https://ollama.com) and share one core.
Each is labelled by who decides the next step: **code** (a chain or a graph, the model fills in
each step) or **the model** (an agent that picks its own tools).

| Feature | What it does | Next step decided by | Built with |
|---|---|---|---|
| **paper2flow** | A research paper in, `overview.pdf` out: a cover, four fact-checked cards and data-flow diagrams | code | LangChain chain |
| **paper2post** | A research paper in, `post.pdf` out: a fact-checked LinkedIn post, its links and the pipeline image | code | LangChain chain |
| **scout** | A topic in, notes on papers found on arXiv out | the model | LangChain `create_agent` |
| **cv2job** | A CV and a job posting in, a tailored CV, a cover letter and a gap report out | code, with one agent step | LangChain chain + `create_agent` |
| **ask** (graph) | A question about a library of PDFs in, a cited answer out | code | LangGraph graph |
| **ask** (agent) | The same task, with the model choosing its tools | the model | LangChain `create_agent` |

**Workflow or agent.** In a workflow the code fixes the order of steps and the model fills each one
in. In an agent the model picks the next action from a set of tools. LangChain chains and LangGraph
graphs are used for both: paper2flow, paper2post and the ask graph are workflows; the ask agent and
scout are agents; cv2job is a workflow whose `gaps` step is an agent.

## Shared core (`labmate.core`)

- **Models:** `ChatOllama` from `langchain-ollama`, built in one place (`core/chat.py`) with the
  configured sampling settings (temperature 0, fixed seed, thinking off).
- **Reply cache:** a SQLite cache serves a repeated request (same messages, model and options)
  without calling the model, so reruns are instant. Its key includes the model, its options and
  the output schema.
- **Structured output:** `structured` validates a reply against a Pydantic schema, puts the schema
  in the system prompt as well as in Ollama's `format` (the MLX builds ignore `format`), and
  retries with the error shown to the model.
- **Grounding:** claim cards carry verbatim quotes that are checked against the source; a
  fact-check loop (judge, rewrite) drops statements the evidence does not support.
- **Tracing:** set `LANGSMITH_TRACING=true` to send chain and graph runs to LangSmith; nothing is
  sent otherwise.

## paper2flow

```
analyze    = ingest | extract | write | factcheck | flows
paper2flow = analyze | render                                          -> overview.pdf
```

`overview.pdf` (A4 portrait):

1. the **cover**: title, authors, date and link, and the paper's overview figure (the one whose
   caption names the architecture, overview, framework, pipeline or model);
2. the **four cards** of a project page, on one page: **Task**, **Challenges**, **Proposed
   method**, **Main results** (every result bullet names the task or dataset its number belongs to);
3. the **end-to-end data flow**, top to bottom, from the raw data to the target output;
4. one page per **detail flow** that breaks a step of the overview down ("detail A", ...).

Every step is a named Runnable that fills in one field of the chain's state and saves it as a JSON
artifact in `runs/<paper id>/`:

| Step | Pattern | What it does |
|---|---|---|
| ingest | plain code | arXiv id or URL, PDF URL or PDF path: sections via the PDF outline, figures cropped, arXiv metadata |
| extract | parallelisation | per section (`.batch`): claim cards with verbatim quotes; quotes not in the paper, or with other numbers, are dropped in code |
| write | prompt chaining | claim cards are grouped into the four cards by their kind (code); one call per card, in parallel; every bullet cites the claim ids it uses |
| factcheck | evaluator-optimizer | numbers and names must be in the cited quotes; a different model judges each bullet; failures are rewritten (at most 2 rounds), then dropped |
| flows | orchestrator-workers | the planner draws the end-to-end flow and picks 1-3 steps; one worker per step draws its detail. Checked in code: starts at inputs, ends at outputs, grounded labels, no invented numbers |
| render | plain code | typed graphs to Mermaid to PNG with mermaid-cli; Typst lays out the PDF without a creation date, so the same inputs render the same bytes |

The model only proposes *data* (typed boxes and arrows); code validates it and writes the
Mermaid, so no model output reaches a diagram or a page unchecked.

## paper2post

```
paper2post = analyze | post | render                                  -> post.pdf
```

Page 1 of `post.pdf` (4:5) is the post text, ready to copy: a hook, 3-5 connected sentences that
tell the paper's story (problem, idea, how it works, result, why it matters), a question about
something specific in the paper, one icon per sentence (a bundled Tabler icon chosen by the
sentence's place in the story) and the links: the paper, its code if it names a repository, and your own links from `config.toml`
`[post.links]`. Page 2 is the end-to-end pipeline diagram, to attach as the post's image. Every
sentence passes the same fact-check as the overview, and the analysis is reused from the cache.

## scout

A research agent: give it a topic, and it searches arXiv, reads, and writes `notes.md` in
`runs/scout/<topic>/`. Nothing fixes the order of its steps; it decides what to do from a few tools.

| Tool | What it does |
|---|---|
| `search_arxiv(query)` | keyword search; it uses several queries |
| `read_abstract(arxiv_id)` | a paper's title and abstract |
| `read_overview(arxiv_id)` | a deep look: runs paper2flow on the paper and returns its fact-checked cards (slow, so limited to `--deep` papers, default 2) |
| `write_notes(notes)` | saves the Markdown notes, refused if they cite an arXiv id the agent has not read |

```bash
make scout TOPIC="linear attention for long sequences"
```

The notes are plain Markdown: Qwen's tool-call format can fail to parse a long argument containing
markup, and Ollama then returns an error; running again usually works.

## cv2job

A CV (YAML) and a job posting (text) in; `cv.pdf`, `cover_letter.pdf` and a separate
`gap_report.pdf` out, in `runs/cv2job/<posting file name>/`.

```
cv2job = requirements | match | gaps | tailor | letter | render
```

| Step | Who decides | What it does |
|---|---|---|
| requirements | model call, checked | reads the posting's requirements (must-have or nice-to-have); every quote must be in the posting, or the model is sent back |
| match | model call per requirement, checked | CV bullets and skills that show it; ids must exist in the CV; empty if the CV shows nothing |
| gaps | **agent** | for requirements nothing matched: searches the CV with other words, asks you a question in the terminal if the CV hints at more, and reports `covered` or `gap`. Your answer is stored verbatim; a `covered` needs CV bullets that exist or your answer |
| tailor | model call per role, checked | picks, orders and rewords up to 4 bullets per role; numbers and names must come from the source bullet, and a tool name such as Kubernetes may not be added |
| letter | model call, checked | up to four highlights (must-haves first), one sentence each from the evidence; everything else is a fixed template |
| render | plain code | three PDFs with Typst |

The cover letter is a fixed template. The years of experience come from the `since` year of a
skill in your CV, never from the model, and there is no claim about the company. The gap report is
for you only: what the CV does not cover (do not claim it), what it covers and by which bullets,
and what you told the agent that is not in your CV yet.

```bash
make cv CV=examples/cv2job/cv.yaml JOB=examples/cv2job/job.txt   # fictional examples
make cv CV=private/cv.yaml JOB=private/job.txt                   # yours (private/ is git-ignored)
```

## ask

Questions about a library of PDFs, answered with citations such as "[Dissertation §4.2, p. 57]".
`library/library.toml` lists the documents:

```toml
[[source]]
id = "dissertation"
file = "pdf/dissertation.pdf"
title = "..."
label = "Dissertation"   # the short name used in citations
```

**The included library** (`library/`) holds the dissertation and ten papers by the repository's author
(PDFs under `library/pdf/`, also on adamfodor.com), so `make index` then `make ask` work after a clone.

**Your own library:** the library is just a folder with a `library.toml` and its PDFs, plus the
`index.sqlite` built from them. Point labmate at any folder with `labmate ask --library DIR ...`
or `LABMATE_LIBRARY=DIR`, for example a private repository next to this one; nothing from it
enters this repository.

**Index** (`make index`): each PDF is split along its outline into sections and sentence-packed
chunks of about 180 words, plus one chunk per figure or table caption. Tables flattened into rows
of numbers are left out, because a model reads their numbers wrongly. Chunks are searched with
BM25 (SQLite FTS5) and dense vectors (`embeddinggemma`), fused with reciprocal rank fusion. Only
the embedding model runs at index time.

**Graph** (`--agent graph`, the default): the code fixes the path.

```mermaid
flowchart LR
    q([question]) --> understand
    understand -->|off topic| abstain
    understand -->|ambiguous| clarify{{"clarify (interrupt)"}}
    understand --> retrieve
    clarify --> retrieve
    retrieve --> grade
    grade -->|not enough| rewrite --> retrieve
    grade -->|enough| answer --> verify --> a([answer])
    grade -->|nothing relevant| abstain --> a
```

- **understand:** decides whether the library can answer the question, and writes the search query.
- **clarify:** if the question could mean different things, the graph pauses (`interrupt`) and
  resumes with the reading you choose (`--choice N`, or asked in the terminal).
- **retrieve, grade, rewrite:** the judge model grades the hits; if they are not enough, the query
  is rewritten and searched again, up to `max_loops` rounds.
- **answer, verify:** the writer answers in sentences that each cite chunk ids; the judge model
  then reads every sentence with its chunks and drops the unsupported ones. Code also drops a
  sentence whose numbers are not in its chunks. If nothing survives, the graph abstains.

**Agent** (`--agent agent`): LangChain's `create_agent` with the tools `search_library`,
`read_context` and `list_sources`. The model decides what to search, whether to read more and when
to stop. Its answer goes through the same sentence checks as the graph's.

```bash
make index                                  # library/*.pdf -> library/index.sqlite
make ask Q="How many heads does the base Transformer use?"
make ask Q="..." AGENT=agent                # the tool-calling agent
make eval-answers                           # graph vs agent on library/golden.yaml
make graphs                                 # Mermaid diagrams of every chain and graph -> docs/graphs.md
make studio                                 # LangGraph Studio (needs a free LangSmith account)
```

## Evaluation

```bash
make eval           # paper2flow metrics of every finished run -> evals/results.md, results.json
make eval-answers   # ask: graph vs agent on library/golden.yaml -> evals/ask/answers.md
```

- **Run metrics** come from the run artifacts, no model needed: verified vs rejected claims, the
  share of first-draft bullets that failed the fact-check, bullets dropped after the loop and the
  size of the diagrams.
- **Answers:** `library/golden.yaml` lists questions, the sources a good answer cites and which
  questions the library cannot answer. The report scores correct refusals, cited sources and how
  many drafted sentences the checks removed, for the graph and the agent.

## Models

| Role | Model |
|---|---|
| Writer / planner | `qwen3.6:35b-mlx` |
| Judge (fact-check, grading, verification) | `gemma4:26b-mlx` |
| Embeddings | `embeddinggemma:latest` |

Both chat models were checked for structured output and tool calling. The roles are set in
`config.toml`.

## Quickstart

```bash
make install     # uv sync + pre-commit hooks + mermaid-cli (needs Node.js)
make check       # lint + type-check + tests + docs (no Ollama needed)

# with Ollama running and the models pulled:
make flow PAPER=1706.03762                 # runs/1706.03762/overview.pdf
make post PAPER=1706.03762                 # runs/1706.03762/post.pdf
make flow PAPER=https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf
```

`LABMATE_OLLAMA_HOST` overrides `[ollama].host`, for Ollama on another machine.

## Reply cache

`[cache]` in `config.toml` sets the SQLite file (`cache/replies.sqlite`) and can switch caching
off. A changed prompt, schema, model or sampling option is a new request and calls the model;
unchanged requests do not. Delete the file to start fresh.

## License

[AGPL-3.0-or-later](LICENSE). PyMuPDF, used for PDF parsing, is AGPL-licensed.
