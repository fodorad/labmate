.PHONY: install check lint format type-check test test-cov docs docs-serve build clean probe probe-fast lock run approve baseline replay eval labels judges trace publish verify site site-serve

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
	uv run mypy paper2carousel

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

# --- Local model tooling (needs a running Ollama) ---

probe:  ## verify model capabilities + benchmark image models -> probe/probe_report.md
	uv run paper2carousel probe --out probe

probe-fast:  ## same, without the (slow) image benchmark
	uv run paper2carousel probe --out probe --skip-images

lock:  ## pin installed model digests into models.lock
	uv run paper2carousel lock

ARXIV ?= 1706.03762
ENGINE ?= plain
PDF ?=
TITLE ?=
# The paper: an arXiv id, or a PDF path/URL (make run PDF=https://.../paper.pdf)
SRC = $(if $(PDF),--pdf "$(PDF)" $(if $(TITLE),--title "$(TITLE)"),$(ARXIV))
REF = $(if $(PDF),$(PDF),$(ARXIV))

run:  ## paper -> outline.yaml for review (reuses finished steps), e.g. make run ARXIV=1706.03762
	uv run paper2carousel run $(SRC) --engine $(ENGINE)

approve:  ## accept the (edited) outline.yaml and finish the carousel
	uv run paper2carousel run $(SRC) --engine $(ENGINE) --approve

baseline:  ## M1 one-shot pipeline, for comparison
	uv run paper2carousel run $(SRC) --baseline

replay:  ## re-run the LLM steps from cassettes only (no Ollama needed)
	uv run paper2carousel run $(SRC) --engine $(ENGINE) --mode replay --fresh --approve

trace:  ## HTML trace viewer -> runs/$(ARXIV)/trace.html
	uv run paper2carousel trace "$(REF)"

# --- Gallery ---

PAPER ?=

publish:  ## copy a finished run + its cassettes into gallery/$(ARXIV)
	uv run paper2carousel publish "$(REF)"

verify:  ## replay gallery entries from their cassettes only and compare every artifact
	uv run paper2carousel verify $(PAPER)

site:  ## build the static gallery -> site/
	uv run paper2carousel site

site-serve: site
	python3 -m http.server 8001 --directory site

# --- Evaluation ---

eval:  ## metrics of all finished runs -> evals/results.md (no Ollama needed)
	uv run paper2carousel eval

labels:  ## blind labelling sheet -> evals/labels.csv; fill the `human` column (s / p / u)
	uv run paper2carousel labels

judges:  ## re-judge your labelled bullets with each judge model -> evals/judges.md (Cohen's kappa)
	uv run paper2carousel judges

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build docs/_build coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
