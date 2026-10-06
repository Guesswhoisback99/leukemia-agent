---
paths:
  - "app/**"
  - "eval/**"
  - "tests/**"
  - "docs/**"
  - "pyproject.toml"
  - "uv.lock"
  - "Dockerfile"
  - "docker-compose.yml"
  - ".env.example"
  - ".gitignore"
---

# Development rules

## Docs
- `docs/requirements/NN-<step>.md`: what a step is, its decisions (and why), eval results. Update it in the
  same change as the code it describes.
- If the requirements conflict with reality (an API changed, a model is unavailable), stop and ask.
- Workarounds get a comment where they are defined, plus a README "Development notes" entry if people need to know.
- When an experiment is settled, delete the losing code path. Git history keeps it.

## Tooling
- Python 3.12, `uv` only (`uv add`, `uv sync`, `uv run`). Never use pip directly.
- Before finishing: `uv run ruff check .`, `uv run ruff format .`, `uv run pytest`.

## Secrets and data
- Never commit `.env`, `data/`, `eval/results/`, `eval/golden.draft.jsonl`. The source PDF is copyrighted.
- AWS credentials come only from the default boto3 chain (`AWS_PROFILE`, `~/.aws`, mounted read-only into
  the container). Never add AWS key fields to settings, `.env.example`, code, or the Dockerfile.

## Tests
- Tests never call AWS, Qdrant, or Docling. Mock or stub them (FastAPI `dependency_overrides`, fake LLMs).
- Logic that can be pure (metadata parsing, formatting, metrics) is pure and unit-tested.
