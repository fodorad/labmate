# labmate pipelines

One diagram per use case, drawn from the code: what goes in, which step runs, who does the work
(code, a model, or an agent), what is checked, and what comes out. The same diagrams are in the
[README](https://github.com/fodorad/labmate#readme); a test fails if the two copies differ or if a
chain step is missing from its diagram. `make graphs` writes LangChain's own drawing of the steps to
[graphs.md](graphs.md).

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

## paper2flow

Type: chain

A paper in, `overview.pdf` out. A chain: the code fixes the order, the model fills in each step.

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

## paper2post

Type: chain

A paper in, `post.pdf` out. It reuses paper2flow's analysis and adds a post.

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

## scout

Type: agent

A topic in, `notes.md` out. An agent: nothing fixes the order of its steps.

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

## cv2job

Type: chain with one agent step

A CV and a job posting in; a tailored CV, a cover letter and a gap report out. A chain whose `gaps` step is an agent.

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

## ask: index

Type: pipeline

Before you can ask, `make index` turns the library's PDFs into a search index. Only the embedding model runs.

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

## ask: graph

Type: graph

A question in, an answer with numbered page citations out. A graph: the code fixes the path. An "off topic" verdict from `understand` is a hint, not a gate: the question is searched once, and the graph refuses only if nothing relevant comes back.

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

## ask: agent

Type: agent

The same task with the model in charge. It searches, reads, and stops when it decides; its answer goes through the same sentence checks as the graph's.

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
