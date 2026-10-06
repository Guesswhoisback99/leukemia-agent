# 02 – Ingest

## Goal
A host command, `uv run python -m app.ingest`, that parses the PDF with Docling, chunks it,
embeds the chunks (dense Titan v2 + sparse BM25), and loads them into Qdrant.

## Context
- Requirements: [`docs/requirements/01-rag-mvp.md`](../../requirements/01-rag-mvp.md)
  (sections: PDF parsing, Chunk content, Vector database).
- Previous milestone: [01 – Skeleton](01-skeleton.md).

## Files
- Create: `app/store.py`, `app/ingest.py`, `tests/test_ingest.py`.

## Steps
1. `app/store.py`: shared factories used by ingest and (later) the chain.
   - `get_embeddings(settings)` → `langchain_aws.BedrockEmbeddings(model_id=..., region_name=...,
     model_kwargs={"dimensions": settings.embedding_dim, "normalize": True})`.
     Check that `model_kwargs` reach the Titan v2 request body.
   - `get_sparse_embeddings()` → `langchain_qdrant.FastEmbedSparse(model_name="Qdrant/bm25")`.
   - `get_vector_store(settings, mode)` → `QdrantVectorStore` with `vector_name="dense"`,
     `sparse_vector_name="sparse"`, `retrieval_mode` mapped from `mode` (`dense` → `RetrievalMode.DENSE`,
     `hybrid` → `RetrievalMode.HYBRID`).
   - `recreate_collection(client, settings)`: delete the collection if it exists, then create it with
     `vectors_config={"dense": VectorParams(size=embedding_dim, distance=Distance.COSINE)}` and
     `sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)}` (IDF is required for BM25).
2. `app/ingest.py`, parsing:
   - `DoclingLoader(file_path=settings.pdf_path, export_type=ExportType.DOC_CHUNKS, chunker=HybridChunker(...))`,
     configured for about 500 tokens per chunk (check the current `docling-core` tokenizer API).
3. Normalize each chunk into a flat structure (pure function, unit-tested):
   - `pages`: sorted unique `page_no` from `metadata["dl_meta"]["doc_items"][*]["prov"][*]`.
   - `page`: first entry of `pages`.
   - `section`: `" > ".join(dl_meta["headings"])`, or `""` when there are none.
   - `has_table`: `True` if any `doc_items[*]["label"]` is `"table"`.
   - `chunk_index`, `source = "protokol_palg_all7.pdf"`.
   - Drop `dl_meta` from the stored metadata (too large for the payload).
4. Heading prefix: check whether `page_content` from the loader already starts with the headings
   (Docling may contextualize chunks). If it doesn't, prepend `section + "\n\n"`. Never add it twice.
5. Exclude a chunk when **all** its `pages` are in `settings.excluded_pages`. Log how many were dropped.
6. Write every final chunk to `data/chunks.jsonl` (`page_content` + metadata) so it can be inspected by hand
   and reused in milestone 04.
7. Load:
   - `recreate_collection(...)`.
   - `get_vector_store(settings, mode="hybrid").add_documents(docs, ids=ids)` so that both dense and sparse vectors
     are written. Point IDs: `uuid5(NAMESPACE_URL, f"{source}:{chunk_index}")`.
   - Batch the calls (e.g. 32 docs per call) to stay within Bedrock rate limits.
8. Log a summary: chunks parsed, excluded, stored, elapsed time.
9. `tests/test_ingest.py` (no Docling, no AWS, no Qdrant): use hand-written fake `dl_meta` dicts to test
   page extraction, section join, page exclusion, idempotent heading prefix, and deterministic IDs.

## Acceptance criteria
- `uv run python -m app.ingest` finishes without errors (the first run downloads Docling models, which is slow).
- The Qdrant dashboard (`http://localhost:6333/dashboard`) shows collection `palg_all7`, and its point count
  matches the number of stored chunks in the log.
- Running ingest again gives the same point count.
- `data/chunks.jsonl`:
  - There is a chunk with `page` 16 whose text contains `METOTREKSAT` and a heading path containing
    `KONSOLIDACJA III`. Its table rows (drug / dose / days) stay readable.
  - No chunk has all of its pages within 1–4.
- `uv run ruff check .` and `uv run pytest` pass.

## Out of scope
- Diagram pages (5, 22, 68), which are a known gap.
- Querying, chain, `/ask`.
- The pypdf parser (milestone 05).
