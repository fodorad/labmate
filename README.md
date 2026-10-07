<p align="center">
  <img src="https://raw.githubusercontent.com/fodorad/labmate/main/docs/_static/logo.svg" alt="labmate logo" width="128" height="128">
</p>

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
| **triage** | A topic and your interests in, arXiv hits sorted into deep read, post or skip out | code, with a decision per paper from the model | LangChain chain |
| **cv2job** | A CV and a job posting in, a tailored CV, a cover letter and a gap report out | code, with one agent step | LangChain chain + `create_agent` |
| **ask** (graph) | A question about a library of PDFs in, a cited answer out | code | LangGraph graph |
| **ask** (agent) | The same task, with the model choosing its tools | the model | LangChain `create_agent` |

**Workflow or agent.** In a workflow the code fixes the order of steps and the model fills each one
in. In an agent the model picks the next action from a set of tools. LangChain chains and LangGraph
graphs are used for both: paper2flow, paper2post and the ask graph are workflows; the ask agent and
scout are agents; cv2job is a workflow whose `gaps` step is an agent.

**Reading the diagrams.** Each use case below has one.

| Shape and colour | Meaning |
|---|---|
| grey box | plain code, no model |
| blue box | a model call (the writer, `[models].text`; in triage the decider, `[models].decider`) |
| purple box | the judge model (`[models].critic`) |
| orange trapezoid | an agent: the model chooses the next action |
| red hexagon | a check in code that can send the model back or drop its output |
| white (rounded or cylinder) | an input, an output or a file |

Every diagram reads top to bottom. Every model call goes through the reply cache
(`cache/replies.sqlite`): an unchanged request is answered from it, so a rerun only calls the model
for what changed.

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

Type: chain

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460, "nodeSpacing": 30, "rankSpacing": 38, "padding": 10}}}%%
flowchart TD
    classDef code fill:#f1efea,stroke:#8a8478,color:#222b35
    classDef llm fill:#e3f2f8,stroke:#4c8aa8,color:#222b35
    classDef judge fill:#efe6f8,stroke:#8a64b0,color:#222b35
    classDef guard fill:#fde8e6,stroke:#c0504d,color:#222b35
    classDef data fill:#fffefd,stroke:#b9b3a8,color:#222b35

    src(["<b>paper</b><br/>arXiv id or URL, PDF URL or PDF file"]):::data
    ingest["<b>ingest</b> · code<br/>download the PDF and the arXiv metadata<br/>sections from the PDF outline<br/>figures cropped with their captions"]:::code
    extract["<b>extract</b> · writer, one call per section<br/>claim cards: a claim and the<br/>verbatim quote that supports it"]:::llm
    quote{{"<b>quote guard</b> · code<br/>the quote must be in the section and carry<br/>the same numbers, else the claim is dropped"}}:::guard
    write["<b>write</b> · code, then writer<br/>code groups the claims by kind into Task,<br/>Challenges, Method and Results; the writer writes<br/>each card in parallel, every bullet cites its claim ids"]:::llm
    factcheck["<b>factcheck</b> · judge model<br/>numbers and names are checked in code,<br/>then the judge rules on every bullet:<br/>supported, partial or unsupported"]:::judge
    rewrite["<b>rewrite</b> · writer<br/>redo the failed bullets, at most 2 rounds,<br/>bullets still failing are dropped"]:::llm
    flows["<b>flows</b> · writer, then one worker per step<br/>plans the end-to-end diagram, picks 1 to 3 steps<br/>to open up, each worker draws its detail"]:::llm
    rules{{"<b>diagram guard</b> · code<br/>starts at inputs, ends at outputs, labels named<br/>in the paper, at most 8 boxes; a detail still<br/>invalid after the retries is dropped"}}:::guard
    render["<b>render</b> · code<br/>typed graph to Mermaid to PNG with mermaid-cli<br/>cover figure: the one whose caption names the architecture<br/>Typst lays out the PDF"]:::code
    out(["<b>runs/id/overview.pdf</b><br/>cover, four cards, data flow, detail pages"]):::data

    src --> ingest --> extract --> quote --> write --> factcheck
    factcheck -->|"bullets that fail"| rewrite --> factcheck
    factcheck -->|"all supported, or rounds used up"| flows --> rules --> render --> out
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

Type: chain

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460, "nodeSpacing": 30, "rankSpacing": 38, "padding": 10}}}%%
flowchart TD
    classDef code fill:#f1efea,stroke:#8a8478,color:#222b35
    classDef llm fill:#e3f2f8,stroke:#4c8aa8,color:#222b35
    classDef judge fill:#efe6f8,stroke:#8a64b0,color:#222b35
    classDef guard fill:#fde8e6,stroke:#c0504d,color:#222b35
    classDef data fill:#fffefd,stroke:#b9b3a8,color:#222b35

    src(["<b>paper</b><br/>arXiv id or URL, PDF URL or PDF file"]):::data
    subgraph analyze ["analyze · the paper2flow chain up to the diagrams (reused from the reply cache if it already ran)"]
        direction LR
        ingest["ingest"]:::code --> extract["extract"]:::llm --> write["write"]:::llm --> factcheck["factcheck"]:::judge --> flows["flows"]:::llm
    end
    post["<b>post</b> · writer<br/>drafts a hook, 3 to 5 sentences that tell the story<br/>(problem, idea, how, result, why it matters)<br/>and a question about something specific in the paper"]:::llm
    shape{{"<b>draft guard</b> · code<br/>every sentence cites known claims<br/>and has no ids in its text"}}:::guard
    recheck["<b>factcheck the sentences</b> · judge model<br/>the same loop as for the cards, 1 rewrite round,<br/>sentences still unsupported are dropped"]:::judge
    render["<b>render</b> · code<br/>icons by place in the story<br/>links: the paper, its code repository if named, yours<br/>the pipeline diagram as the post image"]:::code
    out(["<b>runs/id/post.pdf</b><br/>page 1 the text, page 2 the diagram"]):::data

    src --> analyze --> post --> shape --> recheck --> render --> out
```

Page 1 of `post.pdf` (4:5) is the post text, ready to copy: a hook, 3-5 connected sentences that
tell the paper's story (problem, idea, how it works, result, why it matters), a question about
something specific in the paper, one icon per sentence (a bundled Tabler icon chosen by the
sentence's place in the story) and the links: the paper, its code if it names a repository, and your own links from `config.toml`
`[post.links]`. Page 2 is the end-to-end pipeline diagram, to attach as the post's image. Every
sentence passes the same fact-check as the overview, and the analysis is reused from the cache.

## scout

Type: agent

A research agent: give it a topic, and it searches arXiv, reads, and writes `notes.md` in
`runs/scout/<topic>/`. Nothing fixes the order of its steps; it decides what to do from a few tools.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460, "nodeSpacing": 30, "rankSpacing": 38, "padding": 10}}}%%
flowchart TD
    classDef code fill:#f1efea,stroke:#8a8478,color:#222b35
    classDef agent fill:#ffe9d2,stroke:#c9772b,color:#222b35
    classDef guard fill:#fde8e6,stroke:#c0504d,color:#222b35
    classDef data fill:#fffefd,stroke:#b9b3a8,color:#222b35

    topic(["<b>topic</b><br/>for example linear attention for long sequences"]):::data
    agent[/"<b>agent</b> · the model decides<br/>plans, picks a tool, reads what comes back, repeats<br/>up to 40 steps"\]:::agent
    tools["<b>tools</b> · code<br/>search_arxiv: live arXiv search, several queries<br/>read_abstract: title and abstract of one paper<br/>read_overview: runs paper2flow on one paper and returns<br/>its fact-checked cards (slow, at most --deep papers, default 2)"]:::code
    notes["<b>write_notes</b> · code<br/>saves the Markdown notes"]:::code
    guard{{"<b>citation guard</b> · code<br/>every arXiv id in the notes must have been read with<br/>read_abstract or read_overview, else the notes are refused"}}:::guard
    out(["<b>runs/scout/topic/notes.md</b>"]):::data

    topic --> agent
    agent <-->|"calls"| tools
    agent -->|"writes the notes"| notes --> guard
    guard -->|"refused, the agent fixes them"| agent
    guard -->|"saved"| out
```

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

Type: chain with one agent step

A CV (YAML) and a job posting (text) in; `cv.pdf`, `cover_letter.pdf` and a separate
`gap_report.pdf` out, in `runs/cv2job/<posting file name>/`.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460, "nodeSpacing": 30, "rankSpacing": 38, "padding": 10}}}%%
flowchart TD
    classDef code fill:#f1efea,stroke:#8a8478,color:#222b35
    classDef llm fill:#e3f2f8,stroke:#4c8aa8,color:#222b35
    classDef agent fill:#ffe9d2,stroke:#c9772b,color:#222b35
    classDef guard fill:#fde8e6,stroke:#c0504d,color:#222b35
    classDef data fill:#fffefd,stroke:#b9b3a8,color:#222b35

    inputs(["<b>cv.yaml and job.txt</b><br/>roles with bullet ids and skills with a start year,<br/>and the posting as plain text"]):::data
    requirements["<b>requirements</b> · writer<br/>position, company and the requirements,<br/>each a must-have or a nice-to-have"]:::llm
    quote{{"<b>quote guard</b> · code<br/>each requirement quotes the posting word for word,<br/>else the writer is sent back"}}:::guard
    match["<b>match</b> · writer, one call per requirement<br/>which CV bullets and skills show it,<br/>empty when the CV shows nothing"]:::llm
    ids{{"<b>id guard</b> · code<br/>bullet ids and skill names must exist in the CV"}}:::guard
    open{"requirements<br/>without evidence?"}:::code
    gaps[/"<b>gaps</b> · agent, only for requirements nothing matched<br/>the model decides: search the CV with other words,<br/>read a bullet, ask you a question, report covered or gap"\]:::agent
    tools["<b>tools</b> · code<br/>search_cv, read_cv_item,<br/>ask_candidate (asks you in the terminal), report_finding"]:::code
    honest{{"<b>finding guard</b> · code<br/>covered needs CV bullets that exist or your own answer,<br/>stored word for word; a requirement never reported stays a gap"}}:::guard
    tailor["<b>tailor</b> · writer, one call per role<br/>picks up to 4 bullets, orders them for the job,<br/>rewords them in the posting's terms"]:::llm
    reword{{"<b>reword guard</b> · code<br/>no number, identifier or tool name<br/>that the source bullet does not have"}}:::guard
    letter["<b>letter</b> · writer<br/>up to 4 highlights, must-haves first, one sentence each<br/>years of experience come from the CV, not the model<br/>everything else is a fixed template"]:::llm
    render["<b>render</b> · code<br/>Typst"]:::code
    cvpdf(["<b>cv.pdf</b><br/>the tailored CV"]):::data
    letterpdf(["<b>cover_letter.pdf</b>"]):::data
    gappdf(["<b>gap_report.pdf</b>, for you only<br/>not covered (do not claim), covered and by what,<br/>what you told the agent"]):::data

    inputs --> requirements --> quote --> match --> ids --> open
    open -->|"yes"| gaps
    gaps <-->|"calls"| tools
    gaps --> honest --> tailor
    open -->|"no"| tailor
    tailor --> reword --> letter --> render
    render --> cvpdf
    render --> letterpdf
    render --> gappdf
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

Type: graph and agent

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

### ask: index

Type: pipeline

Run with `make index`. Each PDF is split along its outline into sections and sentence-packed
chunks of about 180 words, plus one chunk per figure or table caption. Tables flattened into rows
of numbers are left out, because a model reads their numbers wrongly. Chunks are searched with
BM25 (SQLite FTS5) and dense vectors (`embeddinggemma`), fused with reciprocal rank fusion. Only
the embedding model runs at index time.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460, "nodeSpacing": 30, "rankSpacing": 38, "padding": 10}}}%%
flowchart TD
    classDef code fill:#f1efea,stroke:#8a8478,color:#222b35
    classDef llm fill:#e3f2f8,stroke:#4c8aa8,color:#222b35
    classDef data fill:#fffefd,stroke:#b9b3a8,color:#222b35

    inputs(["<b>library.toml and the PDFs</b><br/>id, file, title and label per document"]):::data
    sections["<b>sections</b> · code<br/>from the PDF outline, with their numbers, references dropped"]:::code
    chunks["<b>chunks</b> · code<br/>sentence-packed, about 180 words, never across a section,<br/>one chunk per figure or table caption<br/>rows of numbers from tables are left out"]:::code
    embed["<b>embeddings</b> · embedding model<br/>embeddinggemma, one vector per chunk"]:::llm
    out[("<b>index.sqlite</b>, inside the library folder<br/>full-text index (BM25) and vectors")]:::data

    inputs --> sections --> chunks --> embed --> out
```

### ask: graph

Type: graph

The default (`--agent graph`): the code fixes the path.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460, "nodeSpacing": 30, "rankSpacing": 38, "padding": 10}}}%%
flowchart TD
    classDef code fill:#f1efea,stroke:#8a8478,color:#222b35
    classDef llm fill:#e3f2f8,stroke:#4c8aa8,color:#222b35
    classDef judge fill:#efe6f8,stroke:#8a64b0,color:#222b35
    classDef guard fill:#fde8e6,stroke:#c0504d,color:#222b35
    classDef data fill:#fffefd,stroke:#b9b3a8,color:#222b35

    q(["<b>question</b>"]):::data
    understand["<b>understand</b> · writer<br/>a search query, whether it could mean different things,<br/>whether it looks on topic"]:::llm
    clarify["<b>clarify</b> · code, you decide<br/>the graph pauses (interrupt) and you choose one reading"]:::code
    retrieve["<b>retrieve</b> · code<br/>hybrid search in index.sqlite: BM25 and vectors fused (RRF)<br/>skips chunks already judged relevant"]:::code
    grade["<b>grade</b> · judge model<br/>which chunks help, is it enough to answer"]:::judge
    rewrite["<b>rewrite</b> · writer<br/>a new query for what is missing"]:::llm
    answer["<b>answer</b> · writer<br/>sentences, each citing chunk ids"]:::llm
    verify["<b>verify</b> · judge model, then code<br/>the judge reads every sentence with the chunks it cites;<br/>code drops a sentence whose numbers are not in them"]:::judge
    abstain["<b>abstain</b> · code<br/>no sentence survived or nothing relevant: says so instead of guessing"]:::code
    out(["<b>answer</b><br/>sentences with numbered citations,<br/>for example Dissertation §5.4.3, p. 83"]):::data

    q --> understand
    understand -->|"ambiguous"| clarify --> retrieve
    understand -->|"clear"| retrieve
    retrieve --> grade
    grade -->|"not enough, rounds left"| rewrite --> retrieve
    grade -->|"enough, or rounds used up"| answer
    grade -->|"nothing relevant"| abstain
    answer --> verify
    verify -->|"some sentences supported"| out
    verify -->|"none"| abstain
    abstain --> out
```

- **understand:** decides whether the library can answer the question, and writes the search query.
- **clarify:** if the question could mean different things, the graph pauses (`interrupt`) and
  resumes with the reading you choose (`--choice N`, or asked in the terminal).
- **retrieve, grade, rewrite:** the judge model grades the hits; if they are not enough, the query
  is rewritten and searched again, up to `max_loops` rounds.
- **answer, verify:** the writer answers in sentences that each cite chunk ids; the judge model
  then reads every sentence with its chunks and drops the unsupported ones. Code also drops a
  sentence whose numbers are not in its chunks. If nothing survives, the graph abstains.

### ask: agent

Type: agent

Selected with `--agent agent`: LangChain's `create_agent` with the tools `search_library`,
`read_context` and `list_sources`. The model decides what to search, whether to read more and when
to stop. Its answer goes through the same sentence checks as the graph's.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460, "nodeSpacing": 30, "rankSpacing": 38, "padding": 10}}}%%
flowchart TD
    classDef code fill:#f1efea,stroke:#8a8478,color:#222b35
    classDef judge fill:#efe6f8,stroke:#8a64b0,color:#222b35
    classDef agent fill:#ffe9d2,stroke:#c9772b,color:#222b35
    classDef guard fill:#fde8e6,stroke:#c0504d,color:#222b35
    classDef data fill:#fffefd,stroke:#b9b3a8,color:#222b35

    q(["<b>question</b>"]):::data
    agent[/"<b>agent</b> · the model decides<br/>which tool, what to search for, when it has enough<br/>up to 16 steps"\]:::agent
    tools["<b>tools</b> · code<br/>search_library: hybrid search, best chunks with their ids<br/>read_context: a chunk with its neighbours<br/>list_sources: the documents in the library"]:::code
    parse{{"<b>parse</b> · code<br/>split the final text into sentences with their chunk ids in brackets;<br/>a sentence without a valid citation is dropped"}}:::guard
    verify["<b>verify</b> · judge model, then code<br/>the judge reads every sentence with the chunks it cites;<br/>code drops a sentence whose numbers are not in them"]:::judge
    out(["<b>answer</b><br/>numbered citations, or a refusal"]):::data

    q --> agent
    agent <-->|"calls"| tools
    agent -->|"final text"| parse --> verify --> out
```

```bash
make index                                  # library/*.pdf -> library/index.sqlite
make ask Q="How many heads does the base Transformer use?"
make ask Q="..." AGENT=agent                # the tool-calling agent
make eval-answers                           # graph vs agent on library/golden.yaml
make graphs                                 # Mermaid diagrams of every chain and graph -> docs/graphs.md
make studio                                 # LangGraph Studio (needs a free LangSmith account)
```

## triage

Type: workflow with a decision model

A topic and your interests in, a sorted reading list out. The code searches arXiv and loops over the
papers; the decider model (`[models].decider`, by default the decision model `clef-flash`) rates each
paper against your interests, and the code turns the rating into one decision: read it deeply, make a
post of it, or skip it. The code keeps the deep reads within a budget. With `--run` the chosen papers
go through paper2flow or paper2post.

A decision model does not write text. Ollama serves it on `/v1/systemone`: it gets the paper and a
typed question and scores every option in one pass, so a paper costs one short request and the answer
is a number, not a sentence. Any other model (for example `gemma4:e4b`) also works as the decider and
is then asked for the action and a reason as validated JSON. Which kind it is comes from the
capabilities Ollama lists for it. On 36 hand-labelled cases `clef-flash` gets the action right 30
times, `nimble` 29, `tev1` 28 and `gemma4:e4b` 16 (see `docs/benchmarks.md`).

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460, "nodeSpacing": 30, "rankSpacing": 38, "padding": 10}}}%%
flowchart TD
    classDef code fill:#f1efea,stroke:#8a8478,color:#222b35
    classDef llm fill:#e3f2f8,stroke:#4c8aa8,color:#222b35
    classDef guard fill:#fde8e6,stroke:#c0504d,color:#222b35
    classDef data fill:#fffefd,stroke:#b9b3a8,color:#222b35

    topic(["<b>topic</b> and <b>interests</b><br/>for example efficient attention for video"]):::data
    search["<b>search</b> · code<br/>live arXiv search, the first --limit papers (default 8)"]:::code
    decide["<b>rate</b> · decision model, once per paper<br/>scores the title and abstract against the interests on five levels in one pass (/v1/systemone)"]:::llm
    action["<b>action</b> · code<br/>score from 3 deep, from 1.75 post, else skip; relevance 1 to 5 and the level as the reason"]:::code
    budget{{"<b>budget</b> · code<br/>only the --budget most relevant deep papers stay deep (default 2), the rest become skip"}}:::guard
    report(["<b>runs/triage/topic/triage.md</b><br/>and decisions.jsonl"]):::data
    run["<b>--run</b> · code<br/>deep: paper2flow, post: paper2post"]:::code
    out(["<b>runs/id/overview.pdf</b> or <b>post.pdf</b>"]):::data

    topic --> search --> decide --> action --> budget --> report
    budget -.->|"only with --run"| run --> out
```

```bash
make triage TOPIC="efficient attention for video" INTEREST="video emotion recognition"
make triage TOPIC="..." INTERESTS=private/interests.md RUN=1   # also make the PDFs
```

Without `--run` nothing but the table is made, so a triage costs one short request per paper.

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
| Writer / planner (`[models].text`) | `gemma4:26b-mlx` |
| Judge (`[models].critic`) | `gemma4:26b-mlx`, the same model, so only one model is loaded |
| Decider (triage, `[models].decider`) | `clef-flash`, a decision model (10.9 GB); `tev1` (4.5 GB) is the small alternative |
| Embeddings | `embeddinggemma:latest` |

The model is a 16.7 GB mixture of experts (about 4B active parameters), so it reads and writes
faster than the smaller dense models and stays loaded. Ollama lets the GPU use about 26 GB of a
32 GB Mac, and memory used by other programs counts against it: a second large model next to the
first is evicted and loaded again on every call. `ask` runs in an 8k context window
(`[ask].num_ctx`) so its models stay small.

`labmate --profile NAME ...` (or `LABMATE_PROFILE`) picks another pair from `config.toml`:
`mixed` (small `gemma4:e4b` judge), `small` (`gemma4:e4b` only), `fast` (`gemma4:e2b` judge) and
`qwen` (`qwen3.8:27b-mlx` writer). `make bench-micro` measures each model (load, speed, tool
calls, structured output, and how many planted mistakes it catches as a judge) and
`make bench` runs the use cases on each profile; `docs/benchmarks.md` has the tables.

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
