"""Parse the protocol PDF with Docling, chunk it, embed it and load it into Qdrant.

Run on the host: `uv run python -m app.ingest`.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document

from app.config import get_settings
from app.logging_setup import setup_logging
from app.store import get_qdrant_client, get_vector_store, recreate_collection

logger = logging.getLogger(__name__)

SOURCE = "protokol_palg_all7.pdf"
PDF_PATH = Path("data") / SOURCE
CHUNKS_PATH = Path("data/chunks.jsonl")
# Title page + table of contents: the TOC matches almost every query and pushes out real content.
EXCLUDED_PAGES = [1, 2, 3, 4]
# Tokenizer used only to size chunks (Titan's tokenizer is not public); multilingual for Polish.
CHUNK_TOKENIZER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_MAX_TOKENS = 500
# "auto" picks MPS on Macs, which breaks layout detection on Intel Macs with torch 2.2
# (no headings, no tables), so stay on CPU.
DOCLING_DEVICE = "cpu"
BATCH_SIZE = 32


def chunk_pages(dl_meta: dict[str, Any]) -> list[int]:
    return sorted(
        {prov["page_no"] for item in dl_meta.get("doc_items", []) for prov in item.get("prov", [])}
    )


def chunk_section(dl_meta: dict[str, Any]) -> str:
    return " > ".join(dl_meta.get("headings") or [])


def chunk_has_table(dl_meta: dict[str, Any]) -> bool:
    return any(item.get("label") == "table" for item in dl_meta.get("doc_items", []))


def with_heading_prefix(text: str, headings: list[str]) -> str:
    """Make sure the chunk text starts with its heading path, without adding it twice.

    Docling's `contextualize` already prepends headings joined by newlines.
    """
    if not headings:
        return text
    section = " > ".join(headings)
    if text.startswith(("\n".join(headings), section)):
        return text
    return f"{section}\n\n{text}"


def is_excluded(pages: list[int], excluded_pages: list[int]) -> bool:
    return bool(pages) and set(pages) <= set(excluded_pages)


def point_id(source: str, chunk_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{source}:{chunk_index}"))


def normalize_chunk(doc: Document, chunk_index: int) -> Document:
    """Flatten Docling metadata into the payload stored in Qdrant."""
    dl_meta = doc.metadata["dl_meta"]
    pages = chunk_pages(dl_meta)
    headings = dl_meta.get("headings") or []
    return Document(
        page_content=with_heading_prefix(doc.page_content, headings),
        metadata={
            "pages": pages,
            "page": pages[0] if pages else None,
            "section": chunk_section(dl_meta),
            "has_table": chunk_has_table(dl_meta),
            "chunk_index": chunk_index,
            "source": SOURCE,
        },
    )


def select_chunks(docs: list[Document], excluded_pages: list[int]) -> list[Document]:
    return [d for d in docs if not is_excluded(d.metadata["pages"], excluded_pages)]


def load_chunks() -> list[Document]:
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice(DOCLING_DEVICE))
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=CHUNK_TOKENIZER, max_tokens=CHUNK_MAX_TOKENS
    )
    loader = DoclingLoader(
        file_path=PDF_PATH,
        converter=converter,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=tokenizer),
    )
    return [normalize_chunk(doc, i) for i, doc in enumerate(loader.lazy_load())]


def write_jsonl(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            record = {"page_content": doc.page_content, "metadata": doc.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    started = time.perf_counter()

    parsed = load_chunks()
    docs = select_chunks(parsed, EXCLUDED_PAGES)
    logger.info(
        "Parsed %d chunks (%d with tables), excluded %d (pages %s)",
        len(parsed),
        sum(d.metadata["has_table"] for d in parsed),
        len(parsed) - len(docs),
        EXCLUDED_PAGES,
    )
    write_jsonl(docs, CHUNKS_PATH)

    client = get_qdrant_client(settings)
    recreate_collection(client, settings)
    store = get_vector_store(settings, client=client)
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        ids = [point_id(d.metadata["source"], d.metadata["chunk_index"]) for d in batch]
        store.add_documents(batch, ids=ids)
        logger.info("Stored %d/%d chunks", start + len(batch), len(docs))

    stored = client.count(settings.collection_name, exact=True).count
    logger.info(
        "Done: parsed=%d excluded=%d stored=%d collection=%s elapsed=%.1fs",
        len(parsed),
        len(parsed) - len(docs),
        stored,
        settings.collection_name,
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
