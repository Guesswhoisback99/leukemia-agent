# 01 – RAG MVP

RAG layer of the agent for the PALG ALL7 protocol (Polish PDF, 68 pages).

The user asks a question in Polish through a FastAPI endpoint. The system finds relevant
fragments of the protocol in Qdrant and generates a grounded answer with an AWS Bedrock model.
Every answer cites its sources (page + section).

Implementation plans: [`docs/plans/01-rag-mvp/`](../plans/01-rag-mvp/)

## Milestones

- [x] 01 – Skeleton ([plan](../plans/01-rag-mvp/01-skeleton.md))
- [x] 02 – Ingest ([plan](../plans/01-rag-mvp/02-ingest.md))
- [x] 03 – Chain + API ([plan](../plans/01-rag-mvp/03-chain-api.md))
- [x] 04 – Eval: golden set + baseline ([plan](../plans/01-rag-mvp/04-eval.md))
- [x] 05 – Retrieval experiments ([plan](../plans/01-rag-mvp/05-experiments.md))

## Source document

- File: `protokol_palg_all7` – PDF 1.7, 68 pages, Polish, tagged (has a structure tree).
- About 122k characters of text. Clear hierarchy (I/II/III → A/B/C → 1/2/3).
- Most important content is in **dose tables** (drug / dose / days). Naive text extraction flattens them.
- Pages 1–4: title page and table of contents.
- Pages 5, 22, 68: flowchart images with almost no extractable text.

## Decisions

Each decision: what we chose, why, and what we rejected.
Later changes to these architecture decisions are ADRs in [`docs/adr/`](../adr/).

### Orchestration
- **Chose:** LangChain, plain LCEL chain (`retriever | format | prompt | llm`).
- **Why:** widely used library, a single-turn RAG has no branching.
- **Rejected:** hand-rolled boto3 code, LlamaIndex, LangGraph (no loops or decisions yet).

### Embeddings
- **Chose:** `amazon.titan-embed-text-v2:0`, 1024 dims, normalized, cosine distance.
- **Why:** cheapest of the multilingual options, handles Polish.
- **Rejected:** `cohere.embed-v4:0` (about 6× the price), `cohere.embed-multilingual-v3` (512-token input limit).
- Model ID and dimension live in config. Changing either requires re-creating the collection.

### Generation model
- **Chose:** `eu.amazon.nova-2-lite-v1:0` through `ChatBedrockConverse`, temperature 0.
- **Why:** cheap, multilingual, EU-only routing. The Converse API makes swapping models a config change.
- **Rejected:** Anthropic models (excluded by choice), `global.*` profiles (may route outside the EU).
- Comparison candidate in experiments: `openai.gpt-oss-120b-1:0`.

### Region and routing
- **Chose:** `eu-central-1`, only `eu.*` inference profiles or regional on-demand models.

### PDF parsing
- **Chose:** Docling via `langchain-docling`, `ExportType.DOC_CHUNKS` with `HybridChunker` (~500 tokens).
- **Why:** keeps table structure, the heading hierarchy, and **page numbers** in metadata.
- **Rejected:** `PyPDFLoader` (tables become noise; kept only as an experiment baseline),
  PyMuPDF4LLM (AGPL), Docling Markdown export (loses page numbers, which citations need).

### Chunk content
- The section heading path is prepended to each chunk's text before embedding
  (a dose table means little without its heading).
- Pages 1–4 (title + TOC) are excluded. The TOC matches almost every query and pushes out real content.
  The list of excluded pages is configurable.
- Diagram pages (5, 22, 68) are a known gap in v1.

### Vector database
- **Chose:** Qdrant in Docker Compose, port bound to `127.0.0.1`, data in a named volume, no API key.
- Collection has a named dense vector (`dense`) and a named sparse BM25 vector (`sparse`), both filled at ingest.
  The baseline queries dense only. The hybrid experiment just switches the retrieval mode, without re-ingesting.
- **Rejected:** embedded Qdrant (no server), Qdrant Cloud (extra secret).

### Retrieval
- **Baseline:** dense search, top-k = 5.
- Hybrid (dense + BM25) and reranking (`cohere.rerank-v3-5:0`) are measured experiments, not defaults.

### Prompt and answers
- System prompt in Polish. Answer **only** from the provided context.
- Cite after each claim, e.g. `[s. 16, II.C.10]`.
- If the context lacks the answer, reply exactly: `Protokół nie zawiera informacji na ten temat.`
- Never invent doses or extrapolate between patient groups.
- `sources` in the response are built from retrieved chunks, not from what the LLM claims.

### API
- `POST /ask`: request `{ "question": str, "top_k"?: int }` →
  response `{ "answer": str, "sources": [{ "section", "page", "snippet", "score" }], "model": str }`.
- `GET /health`: returns 200 when Qdrant is reachable.
- Single-turn, no streaming, no chat history.

### Runtime
- **Chose:** FastAPI runs as a Docker Compose service next to Qdrant. Ingest and eval run on the host with `uv run`.
- Slim API image: Docling (PyTorch, models) is only in the `ingest` dependency group, which is not installed in the image.
- Inside Compose the API reaches Qdrant at `http://qdrant:6333`. From the host it is `http://localhost:6333`.

### Secrets and config
- `pydantic-settings` loads `.env` (gitignored). `.env.example` is committed.
- `.env` holds only non-secret settings: region, model IDs, Qdrant URL, collection name, PDF path,
  excluded pages, `AWS_PROFILE`, `LOG_LEVEL`.
- AWS keys never enter the project. boto3 uses its default credential chain.
  `~/.aws` is mounted **read-only** into the API container.
- **Rejected:** passing access keys as env vars through `.env`, since one bad ignore rule would commit them.
- Safety net: `.gitignore` + GitHub secret-scanning push protection (verified enabled).
- **Rejected:** pre-commit hooks / gitleaks.
- Optional: a dedicated IAM user limited to `bedrock:InvokeModel*` and `bedrock:Rerank`.

### Source PDF
- Lives in `data/protokol_palg_all7.pdf`. `data/` is gitignored, because the document is copyrighted and the repo is public.

### Observability
- **Chose:** stdout logging only. Per request: question, retrieved pages + scores, model ID, token usage, latency.
  The full prompt is logged only at `LOG_LEVEL=DEBUG`.
- **Rejected for now:** Langfuse (6-container stack), CloudWatch GenAI observability with ADOT.

### Quality
- `ruff` (run by hand) and `pytest`. Tests never call AWS (chain and clients are mocked).
- **Rejected:** CI (GitHub Actions), pre-commit.

### Evaluation
- About 30 Polish golden questions. An LLM drafts them from chunks, then a human reviews them.
- Mix: answers in tables, answers in prose, answers in diagram pages (known gap), questions the protocol cannot answer.
- `eval/golden.jsonl` is committed, with short expected answers (a dose, a day), not long quotes.
- Metrics: hit@k and MRR (by expected page), refusal correctness.
- Results go to `eval/results/` (gitignored).

### Tooling
- Python 3.12 (pinned in `.python-version`), `uv` with dependency groups `api`, `ingest`, `dev`.

## Results

Golden set: `eval/golden.jsonl`, 28 questions (10 table, 10 prose, 3 diagram, 5 unanswerable).
The LLM drafted 22 questions from chunks; the agent checked them against the PDF, dropped 2 (q08: meta
question; q20: the table lost the drug name in parsing), shortened expected answers, and added the diagram and
unanswerable questions; the user approved the set. Page 68 renders blank apart from its heading
("Załącznik nr 3"), so the diagram questions use page 5 (one) and page 22 (two).
Run with `uv run python -m eval.run --label <label>`, k = 5. hit@5 and MRR cover the 23 answerable questions,
so one question moves them by about 0.04. `contains expected` ignores whitespace, case and dash variants
(all rows re-scored with the same metric from saved answers).

| label | hit@5 | MRR | refusal correct | contains expected | notes |
|---|---|---|---|---|---|
| baseline | 0.783 | 0.659 | 0.786 | 0.609 | Dense, top-5, Docling chunks, Nova 2 Lite. hit@5: table 0.90, prose 0.90, diagram 0/3 (known gap). Unanswerable 5/5 refused. Misses: q06 (p.17 not retrieved), q15 (p.46). q09 retrieved at rank 3 but the model refused. A re-run on the final code reproduces retrieval and refusals exactly. |
| hybrid | 0.870 | 0.646 | 0.821 | 0.696 | `RETRIEVAL_MODE=hybrid`. Scores are RRF fusion scores, not cosine. Finds q15 (rank 2, now answered) and diagram d01 (p.5, rank 3), but several first hits drop one rank (q01, q02, q07, q13, q09 3→5), so MRR is slightly lower. |
| rerank | 0.826 | 0.750 | 0.857 | 0.696 | `RERANK_ENABLED=true`, Cohere Rerank 3.5 over 20 dense candidates. Best MRR: q01, q05 move to rank 1, q15 None→1 (answered). Scores are rerank relevance. Answers d03 correctly from p.24 text (pre-phase: dexamethasone + cyclophosphamide), so that diagram fact also exists as prose; hit@5 still counts it as a miss (expected p.22). Cohere quota is 3 req/min: the client uses adaptive retries and a run takes ~10 min. |
| hybrid-rerank | 0.826 | 0.750 | 0.786 | 0.565 | Hybrid candidates + rerank. Same ranks as rerank; d01's hybrid gain does not survive reranking. q19 now refused (table answer lost); lower `contains expected` comes from answer wording on table questions. |
| with-toc | 0.783 | 0.659 | 0.786 | 0.609 | `EXCLUDED_PAGES=[]`, collection `palg_all7_with_toc` (134 points). TOC chunks (p.2–4) enter the top 5 for q06, q13, q15, d01, d03, but only push out chunks below the first hit or misses: no first-hit rank changes. No benefit; keep the exclusion. |
| pypdf | 0.783 | 0.598 | 0.786 | 0.652 | `PARSER=pypdf`, collection `palg_all7_pypdf` (85 chunks, ~2000 chars, no headings). Same hit@5, lower MRR (q07, q10, q12, q14, q18 drop to rank 2). q09 answered, q17 now refused. Tables come out as readable rows on this PDF, so the gap to Docling is smaller than expected. |
| gpt-oss | 0.783 | 0.659 | 0.821 | 0.478 | `LLM_MODEL_ID=openai.gpt-oss-120b-1:0`, same retrieval as baseline. Reasoning blocks are dropped from `answer`. Answers q09, but with the <55 dose (40 mg/m2, p.8) for the >55 group: an extrapolation the prompt forbids. Lower `contains expected` (answers paraphrase, e.g. q17, q19). |

### Summary and proposed default

- **Retrieval:** rerank gives the best ranking (MRR 0.659 → 0.750) and the most correct answers
  (refusal correct 0.857). Hybrid gives the best recall (hit@5 0.870) at no extra cost, and is the only
  setting that reaches a diagram page. Combining them did not add up.
- **Parsing:** Docling stays. pypdf matches hit@5 but ranks worse, and has no section headings for citations.
- **TOC:** keep pages 1–4 excluded (no gain from including them).
- **Model:** keep Nova 2 Lite. gpt-oss-120b extrapolated a dose between age groups (q09), which is the
  failure the prompt rules exist to prevent.
- **Proposed new default** ([ADR 0002](../adr/0002-retrieval-default.md), not applied; needs approval): `RERANK_ENABLED=true` with dense retrieval,
  after raising the Cohere Rerank on-demand quota above 3 requests/minute. Until then, `RETRIEVAL_MODE=hybrid`
  is the safe improvement. With n = 23 answerable questions both gains are 2–4 questions, so grow the golden
  set before relying on the difference between them.

## Backlog (future steps)

- The agent itself, built on this retrieval layer (scope defined in a later requirements file).

- Multimodal descriptions of diagram pages (5, 22, 68) at ingest.
- LLM-as-judge for answer quality.
- Tracing: CloudWatch GenAI observability (ADOT + OpenInference) or Langfuse.
- LangGraph: query rewriting, retry when retrieval is weak.
- Streaming responses, chat history.
