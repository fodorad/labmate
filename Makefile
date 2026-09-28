.PHONY: install check lint format type-check test test-cov docs docs-serve build clean probe lock run approve replay eval labels judges trace publish verify site site-serve index ask eval-rag eval-answers dashboard demo graphs studio

install:
	uv sync --extra dev
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	uv run mypy labmate

test:
	uv run pytest

test-cov:
	uv run pytest --cov-report=html

docs:
	uv run sphinx-build -W -b html docs docs/_build/html

docs-serve: docs
	python3 -m http.server 8000 --directory docs/_build/html

build:
	uv build

check: lint type-check test docs

# --- Shared: local model tooling (needs a running Ollama) ---

probe:  ## verify model capabilities (structured output, tools, determinism) -> probe/probe_report.md
	uv run labmate probe --out probe

lock:  ## pin installed model digests into models.lock
	uv run labmate lock

# --- paper2flow: a paper -> overview.pdf + post.pdf ---

ARXIV ?= 1706.03762
ENGINE ?= plain
PDF ?=
TITLE ?=
# The paper: an arXiv id, or a PDF path/URL (make run PDF=https://.../paper.pdf)
SRC = $(if $(PDF),--pdf "$(PDF)" $(if $(TITLE),--title "$(TITLE)"),$(ARXIV))
REF = $(if $(PDF),$(PDF),$(ARXIV))

run:  ## paper -> outline.yaml for review (reuses finished steps), e.g. make run ARXIV=1706.03762
	uv run labmate paper2flow run $(SRC) --engine $(ENGINE)

approve:  ## accept the (edited) outline.yaml and finish: overview.pdf + post.pdf
	uv run labmate paper2flow run $(SRC) --engine $(ENGINE) --approve

replay:  ## re-run the LLM steps from cassettes only (no Ollama needed)
	uv run labmate paper2flow run $(SRC) --engine $(ENGINE) --mode replay --fresh --approve

trace:  ## HTML trace viewer -> runs/$(ARXIV)/trace.html
	uv run labmate trace "$(REF)"

# --- paper2flow gallery ---

PAPER ?=

publish:  ## copy a finished run + its cassettes into gallery/$(ARXIV)
	uv run labmate paper2flow publish "$(REF)"

verify:  ## replay gallery entries from their cassettes only and compare every artifact
	uv run labmate paper2flow verify $(PAPER)

site:  ## build the static gallery -> site/
	uv run labmate paper2flow site

site-serve: site
	python3 -m http.server 8001 --directory site

# --- paper2flow evaluation ---

eval:  ## metrics of all finished runs -> evals/results.md (no Ollama needed)
	uv run labmate paper2flow eval

labels:  ## blind labelling sheet -> evals/labels.csv; fill the `human` column (s / p / u)
	uv run labmate paper2flow labels

judges:  ## re-judge your labelled bullets with each judge model -> evals/judges.md (Cohen's kappa)
	uv run labmate paper2flow judges

# --- ask: questions about your research (library/library.toml) ---

Q ?= What are the thesis points of the dissertation?
THREAD ?= default
AGENT ?= graph

index:  ## index the library: sections, chunks, summaries, claim cards, embeddings
	uv run labmate ask index

ask:  ## ask a question, e.g. make ask Q="How does BlinkLinMulT differ from LinMulT?" [AGENT=prebuilt]
	uv run labmate ask query "$(Q)" --thread $(THREAD) --agent $(AGENT)

eval-rag:  ## retrieval: recall@k / MRR of BM25, dense, hybrid, hybrid+rewrite -> evals/ask/retrieval.md
	uv run labmate ask eval-retrieval

eval-answers:  ## answers: LangGraph agent vs prebuilt agent on library/golden.yaml -> evals/ask/answers.md
	uv run labmate ask eval-answers

dashboard:  ## live web UI of the agent at http://localhost:8080
	uv run labmate ask dashboard

demo:  ## the dashboard replaying recorded sessions (no Ollama needed)
	uv run labmate ask dashboard --demo

graphs:  ## Mermaid diagrams of every LangGraph graph -> docs/graphs.md
	uv run labmate graphs

studio:  ## open the graphs in LangGraph Studio (langgraph.json)
	uvx --from "langgraph-cli[inmem]" langgraph dev

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build docs/_build coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
