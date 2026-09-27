# Contributing

## Setup

```bash
git clone https://github.com/fodorad/labmate
cd labmate
make install   # uv sync + pre-commit hooks (incl. commit-msg)
```

## Workflow

Feature branches → `dev` → `main`. See the repo's git workflow for the full
branch/PR/CI sequence.

## Test-driven development

Write the test first. Tests should exercise real, useful behavior — no
trivial "does it not crash" placeholders. Target ≥90% coverage
(`make test-cov`). Fixtures for real-data tests live in `tests/fixtures/`
and are only added when the underlying data is actually available.

`tests/` mirrors the `labmate/` source layout (`core/`, `paper2flow/`).

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) — `feat:`,
`fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`, `perf:`,
`build:`. These drive the semantic-versioned changelog via
release-please, so accuracy matters more than brevity.

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
