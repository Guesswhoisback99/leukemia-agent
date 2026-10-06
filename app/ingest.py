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

from app.config import Settings, get_settings
from app.store import get_qdrant_client, get_vector_store, recreate_collection

logger = logging.getLogger(__name__)

SOURCE = "protokol_palg_all7.pdf"
BATCH_SIZE = 32
PYPDF_CHUNK_SIZE = 2000
PYPDF_CHUNK_OVERLAP = 200


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


def normalize_pypdf_chunk(doc: Document, chunk_index: int) -> Document:
    """Flatten a PyPDFLoader chunk; pypdf pages are 0-based."""
    page = doc.metadata["page"] + 1
    return Document(
        page_content=doc.page_content,
        metadata={
            "pages": [page],
            "page": page,
            "section": "",
            "has_table": False,
            "chunk_index": chunk_index,
            "source": SOURCE,
        },
    )


def select_chunks(docs: list[Document], excluded_pages: list[int]) -> list[Document]:
    return [d for d in docs if not is_excluded(d.metadata["pages"], excluded_pages)]


def load_docling_chunks(settings: Settings) -> list[Document]:
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice(settings.docling_device))
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=settings.chunk_tokenizer, max_tokens=settings.chunk_max_tokens
    )
    loader = DoclingLoader(
        file_path=settings.pdf_path,
        converter=converter,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=tokenizer),
    )
    return [normalize_chunk(doc, i) for i, doc in enumerate(loader.lazy_load())]


def load_pypdf_chunks(settings: Settings) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PYPDF_CHUNK_SIZE, chunk_overlap=PYPDF_CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(PyPDFLoader(settings.pdf_path).load())
    return [normalize_pypdf_chunk(doc, i) for i, doc in enumerate(chunks)]


def write_jsonl(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            record = {"page_content": doc.page_content, "metadata": doc.metadata}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True
    )
    settings = get_settings()
    started = time.perf_counter()

    loaders = {"docling": load_docling_chunks, "pypdf": load_pypdf_chunks}
    parsed = loaders[settings.parser](settings)
    docs = select_chunks(parsed, settings.excluded_pages)
    logger.info(
        "Parsed %d chunks (%d with tables), excluded %d (pages %s)",
        len(parsed),
        sum(d.metadata["has_table"] for d in parsed),
        len(parsed) - len(docs),
        settings.excluded_pages,
    )
    write_jsonl(docs, Path(settings.chunks_path))

    client = get_qdrant_client(settings)
    recreate_collection(client, settings)
    store = get_vector_store(settings, mode="hybrid", client=client)
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
