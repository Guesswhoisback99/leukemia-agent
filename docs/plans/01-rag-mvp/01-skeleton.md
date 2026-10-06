# 01 – Skeleton

## Goal
Create the project skeleton: dependencies, config, Docker Compose with Qdrant and an API stub, and ignore rules.

## Context
- Requirements: [`docs/requirements/01-rag-mvp.md`](../../requirements/01-rag-mvp.md)
- Previous milestone: none.

## Files
- Create: `pyproject.toml`, `.python-version`, `uv.lock`, `.gitignore`, `.dockerignore`, `.env.example`,
  `Dockerfile`, `docker-compose.yml`, `app/__init__.py`, `app/config.py`, `app/api.py`,
  `tests/__init__.py`, `tests/test_config.py`, `tests/test_api.py`.
- Move: `protokol_palg_all7` (repo root) → `data/protokol_palg_all7.pdf`.

## Steps
1. Move the PDF to `data/protokol_palg_all7.pdf`. Never commit it.
2. `.python-version` → `3.12`.
3. `pyproject.toml` (project name `leukemia-agent`, `requires-python = ">=3.12,<3.13"`):
   - `[project.dependencies]` (shared): `langchain-core`, `langchain-aws`, `langchain-qdrant`,
     `qdrant-client`, `fastembed`, `pydantic-settings`, `boto3`.
   - `[dependency-groups]`:
     - `api`: `fastapi`, `uvicorn[standard]`
     - `ingest`: `langchain-docling`, `docling`
     - `dev`: `pytest`, `ruff`, `httpx`
   - `[tool.uv] default-groups = ["api", "ingest", "dev"]` (host gets everything).
   - `[tool.ruff]`: `line-length = 100`, `target-version = "py312"`.
   Run `uv lock`.
4. `.gitignore`: `.env`, `data/`, `eval/results/`, `eval/golden.draft.jsonl`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.DS_Store`.
5. `.dockerignore`: everything in `.gitignore` plus `.git/`, `docs/`, `eval/`, `tests/`.
6. `app/config.py`: a `Settings(BaseSettings)` class loading `.env`, with fields and defaults:
   - `aws_region = "eu-central-1"`
   - `embedding_model_id = "amazon.titan-embed-text-v2:0"`, `embedding_dim = 1024`
   - `llm_model_id = "eu.amazon.nova-2-lite-v1:0"`
   - `qdrant_url = "http://localhost:6333"`, `collection_name = "palg_all7"`
   - `pdf_path = "data/protokol_palg_all7.pdf"`
   - `excluded_pages: list[int] = [1, 2, 3, 4]`
   - `top_k = 5`
   - `retrieval_mode: Literal["dense", "hybrid"] = "dense"`
   - `rerank_enabled = False`, `rerank_model_id = "cohere.rerank-v3-5:0"`, `rerank_fetch_k = 20`
   - `log_level = "INFO"`
   - Expose a cached `get_settings()` function.
   - No AWS key fields. `AWS_PROFILE` is read by boto3 directly, not by `Settings`.
7. `.env.example`: every setting above with its default, plus `AWS_PROFILE=default`.
   Add a header comment saying: never put AWS keys here, boto3 reads them from `~/.aws`.
8. `app/api.py`: a FastAPI app with only `GET /health`. It calls Qdrant (`QdrantClient(url).get_collections()`)
   and returns `{"status": "ok"}`, or HTTP 503 when Qdrant is unreachable.
9. `Dockerfile`:
   - Base image `python:3.12-slim`. Copy the `uv` binary from `ghcr.io/astral-sh/uv` (pin a tag).
   - `uv sync --frozen --no-default-groups --group api` (no Docling in the image).
   - Add a non-root user `app` with home `/home/app`.
   - `CMD ["uv", "run", "--no-sync", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]`.
10. `docker-compose.yml`:
    - `qdrant`: image `qdrant/qdrant` (pin a specific version tag),
      ports `127.0.0.1:6333:6333`, named volume `qdrant_data:/qdrant/storage`.
    - `api`: build `.`, `env_file: .env`, `environment: QDRANT_URL=http://qdrant:6333`,
      ports `127.0.0.1:8000:8000`, volume `~/.aws:/home/app/.aws:ro`, `depends_on: qdrant`.
    - `develop.watch`: sync `./app` → `/app/app` and restart, so `docker compose watch` gives hot reload.
11. Tests (no network):
    - `test_config.py`: defaults load, and an env var override works (e.g. `TOP_K=8`).
    - `test_api.py`: `/health` returns 200 with a mocked Qdrant client and 503 when it raises.

## Acceptance criteria
- `uv sync` succeeds. `uv run ruff check .` and `uv run pytest` pass.
- `cp .env.example .env && docker compose up -d --build` starts both services.
- `curl -s localhost:8000/health` → `{"status":"ok"}`.
- `docker compose exec api python -c "import docling"` fails (Docling is not in the image).
- `git status` does not list `.env` or `data/`.

## Out of scope
- Ingest, chain, `/ask` endpoint, eval.
- README and CLAUDE.md content (they already exist).
