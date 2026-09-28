# Contributing

## Setup

```bash
git clone https://github.com/fodorad/labmate
cd labmate
make install   # uv sync + pre-commit hooks (incl. commit-msg)
```

## Workflow

GitHub Flow: `main` is the only long-lived branch. Work on a short-lived `feat/*`,
`fix/*`, `docs/*` or `ci/*` branch, open a PR into `main`, and merge once CI is green.
labmate runs locally, so there is CI but no release or deploy pipeline.

## Test-driven development

Write the test first. Tests should exercise real, useful behavior — no
trivial "does it not crash" placeholders. Target ≥90% coverage
(`make test-cov`). Fixtures for real-data tests live in `tests/fixtures/`
and are only added when the underlying data is actually available.

`tests/` mirrors the `labmate/` source layout (`core/`, `paper2flow/`, `ask/`).

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
