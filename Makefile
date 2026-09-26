.PHONY: install check lint format type-check test test-cov docs docs-serve build clean probe probe-fast lock run approve baseline replay

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

run:  ## paper -> outline.yaml for review (reuses finished steps), e.g. make run ARXIV=1706.03762
	uv run paper2carousel run $(ARXIV)

approve:  ## accept the (edited) outline.yaml and finish the carousel
	uv run paper2carousel run $(ARXIV) --approve

baseline:  ## M1 one-shot pipeline, for comparison
	uv run paper2carousel run $(ARXIV) --baseline

replay:  ## re-run the LLM steps from cassettes only (no Ollama needed)
	uv run paper2carousel run $(ARXIV) --mode replay --fresh --approve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build docs/_build coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
