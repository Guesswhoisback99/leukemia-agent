---
paths:
  - "app/**"
  - "eval/**"
  - "docs/**"
  - "pyproject.toml"
  - "Dockerfile"
  - ".env.example"
---

# Design rules

## AWS
- Region `eu-central-1`. Use only `eu.*` inference profiles or regional models. No `global.*` profiles,
  no Anthropic models.

## Configuration
- Every setting lives in `app/config.py` **and** `.env.example`, with a safe default.
- Model IDs, dimensions, collection name, and retrieval options come from settings, never hard-coded.

## Dependencies
- Dependency groups: `api` (goes into the Docker image), `ingest` (Docling, host only), `dev`.
  Never add Docling or other heavy ingest dependencies to the shared deps or the `api` group.

## Answers
- The refusal phrase is `Protokół nie zawiera informacji na ten temat.`. Keep it in one constant.
- `sources` in API responses come from retrieved chunks, never parsed from LLM output.
