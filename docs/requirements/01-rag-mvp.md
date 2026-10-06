# 01 – RAG MVP

RAG layer of the agent for the PALG ALL7 protocol (Polish PDF, 68 pages). Status: done.

The user asks a question in Polish through `POST /ask`. The system retrieves fragments of the protocol from Qdrant
and generates a grounded answer with an AWS Bedrock model. Every answer cites its sources (page + section).

## Source document

- `data/protokol_palg_all7.pdf` – PDF 1.7, 68 pages, Polish, tagged. Copyrighted: `data/` is gitignored and the repo is public.
- About 122k characters of text. Clear hierarchy (I/II/III → A/B/C → 1/2/3).
- Most important content is in **dose tables** (drug / dose / days). Naive text extraction flattens them.
- Pages 1–4: title page and table of contents.
- Pages 5, 22, 68: flowchart images with almost no extractable text.

## Decisions

### Orchestration
- LangChain LCEL chain: retrieve → format → prompt → LLM, with one branch: empty retrieval returns the refusal
  without calling the LLM. Not LangGraph (no loops or decisions yet), not hand-rolled boto3.

### Models
- **Embeddings:** `amazon.titan-embed-text-v2:0`, 1024 dims, normalized, cosine. Cheapest multilingual option that
  handles Polish (`cohere.embed-v4:0` costs about 6× more; `cohere.embed-multilingual-v3` has a 512-token input limit).
  Changing the model or dimension means re-creating the collection.
- **Generation:** `eu.amazon.nova-2-lite-v1:0` through `ChatBedrockConverse`, temperature 0. Cheap, multilingual,
  EU-only routing. Converse makes swapping models a config change.

### PDF parsing and chunks
- Docling via `langchain-docling`, `ExportType.DOC_CHUNKS` with `HybridChunker` (~500 tokens). It keeps table
  structure, the heading hierarchy and **page numbers**, which citations need. Not Docling Markdown export (loses
  page numbers), not PyMuPDF4LLM (AGPL).
- Each chunk's text starts with its section heading path (a dose table means little without its heading).
- Pages 1–4 (title + TOC) are excluded.
- Diagram pages (5, 22, 68) are a known gap.

### Retrieval
- **Hybrid:** dense Titan + sparse BM25 (FastEmbed `Qdrant/bm25`) fused with RRF, top-k 5. Chosen from the
  experiments below. `score` in sources is an RRF fusion score, not cosine similarity.
- Qdrant in Docker Compose: port bound to `127.0.0.1`, named volume, no API key. Collection with named vectors
  `dense` and `sparse`. Not embedded Qdrant (no server), not Qdrant Cloud (extra secret).

### Prompt and answers
- Polish system prompt. Answer **only** from the provided context; cite after each claim, e.g.
  `[s. 16, 10. KONSOLIDACJA III]`; never invent doses or extrapolate between patient groups.
- If the context lacks the answer, reply exactly `Protokół nie zawiera informacji na ten temat.`

### API
- `POST /ask`: `{ "question": str, "top_k"?: int }` →
  `{ "answer": str, "sources": [{ "section", "page", "snippet", "score" }], "model": str }`.
- `GET /health`: 200 when Qdrant is reachable.
- Single-turn, no streaming, no chat history.

### Runtime and secrets
- The API runs in Docker Compose next to Qdrant; ingest and eval run on the host with `uv run`. Docling (PyTorch,
  models) is only in the `ingest` dependency group, so the API image stays slim.
- AWS keys never enter the project: boto3 uses its default chain, `~/.aws` is mounted read-only. Keys in `.env`
  were rejected because one bad ignore rule would commit them. Safety net: `.gitignore` + GitHub secret-scanning
  push protection; no pre-commit hooks.

### Observability and quality
- stdout logging only. Per request: question, retrieved pages + scores, model ID, token usage, latency.
  The full prompt only at `LOG_LEVEL=DEBUG`. Langfuse and CloudWatch GenAI observability are backlog.
- `ruff` (by hand) and `pytest`. No CI.

### Evaluation
- About 30 Polish golden questions in `eval/golden.jsonl` (committed): an LLM drafts them from chunks
  (`eval/generate_golden.py`), a human reviews them. Short expected answers (a dose, a day), not long quotes.
- Mix: answers in tables, in prose, in diagram pages (known gap), and questions the protocol cannot answer.
- Metrics: hit@k and MRR (by expected page), refusal correctness, `contains expected`.
  Results go to `eval/results/` (gitignored).

## Results

Golden set: `eval/golden.jsonl`, 28 questions (10 table, 10 prose, 3 diagram, 5 unanswerable).
The LLM drafted 22 questions from chunks; the agent checked them against the PDF, dropped 2 (q08: meta
question; q20: the table lost the drug name in parsing), shortened expected answers, and added the diagram and
unanswerable questions; the user approved the set. Page 68 renders blank apart from its heading
("Załącznik nr 3"), so the diagram questions use page 5 (one) and page 22 (two).
Run with `uv run python -m eval.run --label <label>`, k = 5. hit@5 and MRR cover the 23 answerable questions,
so one question moves them by about 0.04. `contains expected` ignores whitespace, case and dash variants
(all rows re-scored with the same metric from saved answers).

The experiment switches in the notes (`RETRIEVAL_MODE`, `RERANK_ENABLED`, `PARSER`, `EXCLUDED_PAGES`) were removed
once the experiments were settled. To reproduce a row, check out commit `bad7c2b`.

| label | hit@5 | MRR | refusal correct | contains expected | notes |
|---|---|---|---|---|---|
| baseline | 0.783 | 0.659 | 0.786 | 0.609 | Dense, top-5, Docling chunks, Nova 2 Lite. hit@5: table 0.90, prose 0.90, diagram 0/3 (known gap). Unanswerable 5/5 refused. Misses: q06 (p.17 not retrieved), q15 (p.46). q09 retrieved at rank 3 but the model refused. A re-run on the final code reproduces retrieval and refusals exactly. |
| hybrid | 0.870 | 0.646 | 0.821 | 0.696 | `RETRIEVAL_MODE=hybrid`. Scores are RRF fusion scores, not cosine. Finds q15 (rank 2, now answered) and diagram d01 (p.5, rank 3), but several first hits drop one rank (q01, q02, q07, q13, q09 3→5), so MRR is slightly lower. |
| rerank | 0.826 | 0.750 | 0.857 | 0.696 | `RERANK_ENABLED=true`, Cohere Rerank 3.5 over 20 dense candidates. Best MRR: q01, q05 move to rank 1, q15 None→1 (answered). Scores are rerank relevance. Answers d03 correctly from p.24 text (pre-phase: dexamethasone + cyclophosphamide), so that diagram fact also exists as prose; hit@5 still counts it as a miss (expected p.22). Cohere quota is 3 req/min: the client uses adaptive retries and a run takes ~10 min. |
| hybrid-rerank | 0.826 | 0.750 | 0.786 | 0.565 | Hybrid candidates + rerank. Same ranks as rerank; d01's hybrid gain does not survive reranking. q19 now refused (table answer lost); lower `contains expected` comes from answer wording on table questions. |
| with-toc | 0.783 | 0.659 | 0.786 | 0.609 | `EXCLUDED_PAGES=[]`, collection `palg_all7_with_toc` (134 points). TOC chunks (p.2–4) enter the top 5 for q06, q13, q15, d01, d03, but only push out chunks below the first hit or misses: no first-hit rank changes. No benefit; keep the exclusion. |
| pypdf | 0.783 | 0.598 | 0.786 | 0.652 | `PARSER=pypdf`, collection `palg_all7_pypdf` (85 chunks, ~2000 chars, no headings). Same hit@5, lower MRR (q07, q10, q12, q14, q18 drop to rank 2). q09 answered, q17 now refused. Tables come out as readable rows on this PDF, so the gap to Docling is smaller than expected. |
| gpt-oss | 0.783 | 0.659 | 0.821 | 0.478 | `LLM_MODEL_ID=openai.gpt-oss-120b-1:0`, same retrieval as baseline. Reasoning blocks are dropped from `answer`. Answers q09, but with the <55 dose (40 mg/m2, p.8) for the >55 group: an extrapolation the prompt forbids. Lower `contains expected` (answers paraphrase, e.g. q17, q19). |

### Conclusions

- **Retrieval: hybrid is the default.** Best recall (hit@5 0.870) at no extra cost, and the only setting that
  reaches a diagram page. Rerank ranks best (MRR 0.750) but the Cohere Rerank on-demand quota is 3 requests/minute,
  too low for the API, so its code was removed (see backlog). Combining them did not add up.
- **Parsing:** Docling stays. pypdf matches hit@5 but ranks worse, and has no section headings for citations.
- **TOC:** keep pages 1–4 excluded (no gain from including them).
- **Model:** keep Nova 2 Lite. gpt-oss-120b extrapolated a dose between age groups (q09), which is the
  failure the prompt rules exist to prevent.
- With n = 23 answerable questions each gain is 2–4 questions. Grow the golden set before relying on small differences.

## Backlog

- The agent itself, built on this retrieval layer (scope in a later requirements file).
- Rerank (`cohere.rerank-v3-5:0` via `BedrockRerank`) once the quota is raised; the code is in `bad7c2b`.
- Multimodal descriptions of diagram pages (5, 22, 68) at ingest.
- Grow the golden set; LLM-as-judge for answer quality.
- Tracing: CloudWatch GenAI observability (ADOT + OpenInference) or Langfuse.
- LangGraph: query rewriting, retry when retrieval is weak.
- Streaming responses, chat history.
