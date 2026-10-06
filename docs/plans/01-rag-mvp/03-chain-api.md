# 03 – Chain + API

## Goal
An LCEL RAG chain (retrieve → format → prompt → Nova 2 Lite) exposed as `POST /ask`, with request logging.

## Context
- Requirements: [`docs/requirements/01-rag-mvp.md`](../../requirements/01-rag-mvp.md)
  (sections: Prompt and answers, API, Observability).
- Previous milestone: [02 – Ingest](02-ingest.md). The collection must already be loaded.

## Files
- Create: `app/chain.py`, `app/logging_setup.py`, `tests/test_chain.py`.
- Change: `app/api.py`, `tests/test_api.py`.

## Steps
1. `app/chain.py`:
   - `REFUSAL = "Protokół nie zawiera informacji na ten temat."`
   - `retrieve(question, top_k) -> list[tuple[Document, float]]`: calls `similarity_search_with_score` on
     `get_vector_store(settings, mode=settings.retrieval_mode)`. Scores are needed, so don't use `as_retriever()`.
   - `format_context(docs)`: each chunk as `[s. {page}, {section}]\n{page_content}`, separated by `\n\n---\n\n`.
   - Prompt (`ChatPromptTemplate`, system + human), written in Polish. The system message says:
     answer only from the context, cite after each claim using the label shown above the chunk
     (format `[s. <strona>, <sekcja>]`), never invent or extrapolate doses between patient groups,
     and if the context does not contain the answer, reply with exactly `REFUSAL` and nothing else.
   - LLM: `langchain_aws.ChatBedrockConverse(model=settings.llm_model_id, region_name=..., temperature=0)`.
   - Compose it with LCEL. The result must carry the answer text, the retrieved `(doc, score)` pairs,
     and `usage_metadata` from the `AIMessage`
     (e.g. `RunnablePassthrough.assign(...)` over `{"question", "top_k"}`).
   - If retrieval returns nothing, return `REFUSAL` without calling the LLM.
   - `build_chain(settings)` builds clients once. No module-level AWS calls at import time.
2. `app/logging_setup.py`: configure root logging from `settings.log_level`, one line per record to stdout.
3. `app/api.py`:
   - Build the chain once in the FastAPI `lifespan`. Expose it through a dependency (`get_chain`) so tests can override it.
   - `POST /ask`:
     - Request model: `question: str` (1–1000 chars), `top_k: int | None` (1–20, default `settings.top_k`).
     - Response model: `answer: str`, `sources: list[Source]`, `model: str`.
       `Source = {section: str, page: int, snippet: str, score: float}`. `snippet` is the first ~300 chars of the chunk.
     - `sources` come from the retrieved chunks, never parsed from the LLM output.
     - Bedrock/Qdrant errors → log the exception, return HTTP 502 with a generic message.
   - Keep `GET /health` unchanged.
4. Per-request log line (INFO): question, retrieved `page`/`score` pairs, model ID, input/output tokens, latency ms.
   At DEBUG, also log the full rendered prompt.
5. Tests (no AWS, no Qdrant):
   - `test_chain.py`: `format_context` output; empty retrieval returns `REFUSAL` and never calls the LLM
     (fake LLM via `langchain_core` fakes or a stub).
   - `test_api.py`: `/ask` with an overridden `get_chain` returns the response shape; empty question → 422;
     `top_k=50` → 422; a chain raising → 502.

## Acceptance criteria
- `docker compose up -d --build` then:
  ```
  curl -s localhost:8000/ask -H 'content-type: application/json' \
    -d '{"question": "Jaka jest dawka metotreksatu w konsolidacji III u chorych poniżej 55 r.ż.?"}'
  ```
  The answer contains `1500 mg/m2` with a citation, and `sources` includes page 16.
- Asking something the protocol does not cover (e.g. `"Jaka jest stolica Australii?"`) returns exactly `REFUSAL`.
- `docker compose logs api` shows one log line per request with pages, scores, tokens, and latency.
  With `LOG_LEVEL=DEBUG` the prompt is also logged.
- `uv run ruff check .` and `uv run pytest` pass.

## Out of scope
- Hybrid search, reranking (milestone 05).
- Streaming, chat history, tracing backends.
