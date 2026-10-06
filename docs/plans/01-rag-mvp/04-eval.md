# 04 – Eval: golden set + baseline

## Goal
A reviewed golden set of about 30 Polish questions and an eval script that measures retrieval and refusal
quality. Record the baseline numbers.

## Context
- Requirements: [`docs/requirements/01-rag-mvp.md`](../../requirements/01-rag-mvp.md) (section: Evaluation).
- Previous milestone: [03 – Chain + API](03-chain-api.md). Uses `data/chunks.jsonl` from [02 – Ingest](02-ingest.md).

## Files
- Create: `eval/__init__.py`, `eval/generate_golden.py`, `eval/metrics.py`, `eval/run.py`,
  `eval/golden.jsonl` (after human review), `tests/test_metrics.py`.
- Change: `docs/requirements/01-rag-mvp.md` (add a `## Results` section).

## Golden record format
One JSON object per line:
```json
{"id": "q01", "question": "...", "expected_answer": "1500 mg/m2", "expected_pages": [16], "type": "table"}
```
`type` is one of `table`, `prose`, `diagram`, `unanswerable`. Unanswerable questions have `expected_pages: []`
and `expected_answer` equal to the refusal phrase.

## Steps
1. `eval/generate_golden.py` (host, `uv run python -m eval.generate_golden`):
   - Read `data/chunks.jsonl`. Sample about 12 chunks with `has_table=True` and about 10 without, spread across
     the document (fixed random seed).
   - For each chunk, ask the LLM (`ChatBedrockConverse`, model from settings) via `with_structured_output`
     for one Polish question that the chunk answers, plus a **short** expected answer (a dose, a day, a name;
     at most ~15 words, no long quotes).
   - `expected_pages` = the chunk's `pages`. `type` = `table` or `prose`.
   - Write to `eval/golden.draft.jsonl` (gitignore this file) and print a reminder to review it.
2. Human review (the user does this, not the agent): fix or drop bad questions, then add by hand
   - 3 `diagram` questions about pages 5, 22, 68 (expected to fail in v1, which documents the gap);
   - 5 `unanswerable` questions: 2 out of domain, 3 near the domain but not in the protocol.

   Save the result as `eval/golden.jsonl` (committed).
3. `eval/metrics.py` (pure functions, unit-tested):
   - `first_hit_rank(retrieved_pages_per_rank, expected_pages) -> int | None`. A rank counts as a hit when the
     chunk's `pages` intersect `expected_pages`.
   - `hit_at_k`, `mrr` over a list of ranks.
   - `refusal_correct(answer, type)`: for `unanswerable`, `answer.strip() == REFUSAL`; otherwise `answer` must not be the refusal.
   - `contains_expected(answer, expected_answer)`: case- and whitespace-normalized substring check
     (cheap signal only).
4. `eval/run.py` (`uv run python -m eval.run --label baseline [--k 5] [--retrieval-only]`):
   - Use `retrieve` and `build_chain` from `app.chain` directly (not over HTTP), with settings from `.env`.
   - For each question: ranks for hit@k/MRR (skip `unanswerable`), then (unless `--retrieval-only`) the
     full answer for refusal correctness and `contains_expected`.
   - Aggregate overall and per `type`. Print a table.
   - Save `eval/results/<timestamp>-<label>.json` with: label, settings snapshot, aggregates, and per-question rows
     (question, ranks, retrieved pages, answer).
5. `tests/test_metrics.py`: cover all functions in `eval/metrics.py`, including edge cases (no hit, empty
   `expected_pages`, multi-page chunks).
6. Run `--label baseline` and add a `## Results` section to `docs/requirements/01-rag-mvp.md` with a table:
   `label | hit@5 | MRR | refusal correct | contains expected | notes`.

## Acceptance criteria
- `eval/golden.jsonl` exists, was reviewed by the user, and has ≥ 25 records covering all four types.
- `uv run python -m eval.run --label baseline` prints the table and writes a results file.
- `eval/results/` and `eval/golden.draft.jsonl` are not listed by `git status`.
- The baseline row is in the requirements doc.
- `uv run ruff check .` and `uv run pytest` pass.

## Out of scope
- LLM-as-judge, Ragas.
- Retrieval changes (milestone 05).
