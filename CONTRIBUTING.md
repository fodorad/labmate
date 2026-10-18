# Contributing

## Setup

```bash
git clone https://github.com/fodorad/labmate
cd labmate
make install   # uv sync + pre-commit hooks (incl. commit-msg)
```

## Workflow

GitHub Flow: `main` is the only long-lived branch. Work on a short-lived `feat/*`,
`fix/*`, `docs/*` or `ci/*` branch, open a PR into `main`, and merge once `make check`
passes locally. The GitHub Actions workflow is manual-only for now; labmate runs
locally, so there is no release or deploy pipeline.

## Test-driven development

Write the test first. Each test states a behaviour a feature relies on ("an abstaining
answer says why"); no tests of wire formats, private helpers
already covered by behaviour tests, or constants. Never patch production code: pass the
dependency in (an argument, a config value, the fake Ollama in `tests/conftest.py`).
Target ≥90% coverage (`make test-cov`).

`tests/` mirrors the `labmate/` source layout: `labmate/ask/index.py` is tested in
`tests/ask/test_index.py`.

## Adding a use case

A use case is one package under `labmate/` with the same shape as the others:

- the pipeline (`chain.py` for a LangChain chain, `graph.py` for a LangGraph graph, `agent.py`
  for a tool-calling agent), with its prompts as Markdown files in `prompts/`;
- `cli.py`: `add_parser(sub)` adds its command and sets `handler`, the function that runs it with
  `(config, args, ctx)`. Import the pipeline inside that function, so `labmate --help` works
  without the `[ask]` extra.

Then add the package name to `FEATURES` in `labmate/cli.py` and a target to the `Makefile`.
Shared plumbing lives in `labmate/core/`: chat models and the reply cache, the Ollama server,
arXiv and PDF ingest. To benchmark it, add a runner to `labmate/bench/macro.py`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) — `feat:`,
`fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`, `perf:`,
`build:`. The history is the changelog, so accuracy matters more than brevity.

## No AI co-author trailers

Commits must not carry AI co-author trailers or AI attribution. A `commit-msg`
pre-commit hook rejects them.

## Docstrings

Google-style docstrings, including on module-level attributes and
constants (Sphinx autodoc otherwise renders an empty table for them).

## Before opening a PR

```bash
make check   # lint + type-check + test + docs
```
