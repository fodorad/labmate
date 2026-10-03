.PHONY: install check lint format type-check test test-cov docs docs-serve build clean flow post eval index ask eval-answers graphs studio

install:
	uv sync --extra dev
	uv run pre-commit install
	npx --yes @mermaid-js/mermaid-cli@12.0.0 --version  # diagram renderer (needs Node.js)

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

# --- paper2flow and paper2post: a paper -> overview.pdf / post.pdf ---

PAPER ?= 1706.03762
# The paper: an arXiv id or URL, a PDF URL or a PDF path

flow:  ## a paper -> runs/<id>/overview.pdf, e.g. make flow PAPER=1706.03762
	uv run labmate paper2flow "$(PAPER)"

post:  ## a paper -> runs/<id>/post.pdf (reuses the overview's analysis)
	uv run labmate paper2post "$(PAPER)"

# --- evaluation of the paper chains ---

eval:  ## metrics of all finished runs -> evals/results.md (no Ollama needed)
	uv run labmate eval

# --- ask: questions about a library of documents (library/library.toml) ---

Q ?= What does the library say about linear attention?
AGENT ?= graph

index:  ## index the library: sections, chunks, embeddings
	uv run labmate ask index

ask:  ## ask a question, e.g. make ask Q="How does BlinkLinMulT differ from LinMulT?" [AGENT=agent]
	uv run labmate ask query "$(Q)" --agent $(AGENT)

eval-answers:  ## answers: graph vs agent on library/golden.yaml -> evals/ask/answers.md
	uv run labmate ask eval-answers

graphs:  ## Mermaid diagrams of the chains and graphs -> docs/graphs.md
	uv run labmate graphs

studio:  ## open the graphs in LangGraph Studio (langgraph.json)
	uvx --from "langgraph-cli[inmem]" langgraph dev

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build docs/_build coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
