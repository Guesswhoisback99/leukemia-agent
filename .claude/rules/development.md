---
paths:
  - "app/**"
  - "eval/**"
  - "tests/**"
  - "docs/plans/**"
  - "pyproject.toml"
  - "uv.lock"
  - "Dockerfile"
  - "docker-compose.yml"
  - ".env.example"
  - ".gitignore"
---

# Development rules

## How work is organized
- `docs/requirements/NN-<step>.md`: what a step is, decisions (and why), milestone checklist.
- `docs/plans/NN-<step>/MM-<milestone>.md`: exact steps for one milestone.
- Work on **one plan file at a time**. Follow its steps, meet its acceptance criteria, respect its "Out of scope".
- When a milestone is done, tick it in the matching requirements file.
- If a plan conflicts with the requirements or with reality (an API changed, a model is unavailable), stop and ask.
  Don't silently pick something else.
- Record architecturally significant decisions (components, technologies, interfaces, data model, deployment,
  anything expensive to reverse) as an ADR in `docs/adr/NNNN-<slug>.md`, indexed in `docs/adr/README.md`.
  Commit it with the change it explains. Never rewrite an accepted ADR; supersede it.
- Other decisions are not ADRs: workarounds and config defaults get a comment where they are defined
  (and a README "Development notes" entry if people need to know); eval findings go in the requirements "Results".

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
