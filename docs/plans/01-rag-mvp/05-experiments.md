# 05 – Retrieval experiments

## Goal
Measure five changes against the baseline with `eval/run.py` and record each as a row in the results table.
Defaults change only after the user decides.

## Context
- Requirements: [`docs/requirements/01-rag-mvp.md`](../../requirements/01-rag-mvp.md)
  (sections: Retrieval, PDF parsing, Generation model, Results).
- Previous milestone: [04 – Eval](04-eval.md). The baseline row must exist.

## Files
- Change: `app/config.py`, `.env.example`, `app/chain.py`, `app/ingest.py`, `pyproject.toml`,
  `tests/test_chain.py`, `tests/test_ingest.py`, `docs/requirements/01-rag-mvp.md`.

## Rules for every experiment
- Change **one** variable per run. Everything else stays at baseline.
- Switch experiments through config (`.env` or env vars), not by editing code between runs.
- When an experiment needs different chunks, ingest into a **separate collection** (`COLLECTION_NAME=...`)
  so the baseline collection stays intact.
- Run `uv run python -m eval.run --label <name>` and add one row to `## Results` with a short note
  (what changed, notable wins and losses per question `type`).

## Steps
1. **E1 – hybrid search** (`--label hybrid`)
   - `RETRIEVAL_MODE=hybrid`. No code change expected: sparse vectors already exist (milestone 02).
   - Note in the results that scores are fusion (RRF) scores now, not cosine similarities.
2. **E2 – rerank** (`--label rerank`, and `--label hybrid-rerank` if E1 helped)
   - In `app/chain.py`: when `settings.rerank_enabled`, retrieve `settings.rerank_fetch_k` chunks, then rerank with
     `langchain_aws` `BedrockRerank` (`model_arn=f"arn:aws:bedrock:{region}::foundation-model/{rerank_model_id}"`)
     and keep `top_k`. `score` in sources becomes the rerank relevance score.
   - Unit test with a stubbed reranker: order and `top_k` cut are correct, and the reranker is not called when disabled.
3. **E3 – keep the TOC** (`--label with-toc`)
   - `EXCLUDED_PAGES=[]`, `COLLECTION_NAME=palg_all7_with_toc`. Re-ingest, then run eval against that collection.
4. **E4 – pypdf baseline** (`--label pypdf`)
   - Add `parser: Literal["docling", "pypdf"] = "docling"` to settings.
   - Add `pypdf`, `langchain-community`, `langchain-text-splitters` to the `ingest` group.
   - `pypdf` path in `app/ingest.py`: `PyPDFLoader` + `RecursiveCharacterTextSplitter`
     (about 2000 chars, 200 overlap). `page` = loader page + 1 (pypdf pages are 0-based), `pages = [page]`,
     `section = ""`, `has_table = False`. Same page exclusion and ID scheme.
   - `PARSER=pypdf`, `COLLECTION_NAME=palg_all7_pypdf`. Re-ingest and run eval.
   - Unit test for the page-number shift.
5. **E5 – gpt-oss-120b** (`--label gpt-oss`)
   - `LLM_MODEL_ID=openai.gpt-oss-120b-1:0`. Retrieval metrics should match baseline. Compare refusal correctness
     and `contains_expected`.
   - If the model returns reasoning blocks through Converse, make sure only the final text becomes `answer`.
6. Summarize under the results table: which settings look best, and a proposed new default config.
   **Do not change defaults** in `app/config.py` / `.env.example` until the user approves.

## Acceptance criteria
- `## Results` has rows for baseline, hybrid, rerank (and hybrid-rerank if run), with-toc, pypdf, gpt-oss.
- The baseline collection `palg_all7` still exists and is unchanged.
- With all new settings at their defaults, behavior equals the baseline (same eval numbers within noise).
- `uv run ruff check .` and `uv run pytest` pass.

## Out of scope
- Diagram-page handling, LLM-as-judge, tracing, LangGraph (backlog).
