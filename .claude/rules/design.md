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
- `Settings` in `app/config.py` holds only what changes per environment or experiment: model IDs, region,
  Qdrant URL, collection, top-k, log level. Fixed values are module constants with a comment saying why.
- `.env.example` lists only what a user must set, plus commented-out common overrides.

## Dependencies
- Dependency groups: `api` (goes into the Docker image), `ingest` (Docling, host only), `dev`.
  Never add Docling or other heavy ingest dependencies to the shared deps or the `api` group.

## Answers
- The refusal phrase is `Protokół nie zawiera informacji na ten temat.`. Keep it in one constant.
- `sources` in API responses come from retrieved chunks, never parsed from LLM output.
