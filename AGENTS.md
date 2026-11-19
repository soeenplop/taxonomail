# AGENTS.md

Instructions for coding agents working on taxonomail.

- Everything in English: code, comments, docs, commit messages, PRs.
- This repository is public. Never commit real mail or anything derived from it (output of
  a run, taxonomies, cluster names, prompts or LLM answers about real mail). Tests use
  synthetic mail written for the test.
- Secrets live in `.env` (gitignored) or the environment, never in the repository.
- Commits follow Conventional Commits: `<type>(<scope>): <summary>`.
- Python 3.12, `uv`, `typer`, `pydantic` v2. Keep it small: the tool reads mail, clusters
  it with BERTopic, and lets an LLM name and merge the clusters.

Checks before every commit:

```sh
uv sync --frozen
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

Tests must run without network, API keys or model downloads: use the fakes in
`tests/fakes.py`.
