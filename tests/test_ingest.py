import json
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.config import Settings
from app.ingest import (
    chunk_has_table,
    chunk_pages,
    chunk_section,
    is_excluded,
    normalize_chunk,
    point_id,
    select_chunks,
    with_heading_prefix,
)
from app.store import get_embeddings

DL_META = {
    "headings": ["II. LECZENIE", "10. KONSOLIDACJA III"],
    "doc_items": [
        {"label": "text", "prov": [{"page_no": 17}]},
        {"label": "table", "prov": [{"page_no": 16}, {"page_no": 17}]},
    ],
}


def test_chunk_pages_sorted_unique():
    assert chunk_pages(DL_META) == [16, 17]
    assert chunk_pages({"doc_items": []}) == []


def test_chunk_section():
    assert chunk_section(DL_META) == "II. LECZENIE > 10. KONSOLIDACJA III"
    assert chunk_section({"doc_items": []}) == ""
    assert chunk_section({"headings": None}) == ""


def test_chunk_has_table():
    assert chunk_has_table(DL_META) is True
    assert chunk_has_table({"doc_items": [{"label": "text", "prov": []}]}) is False


def test_heading_prefix_added_once():
    headings = DL_META["headings"]
    once = with_heading_prefix("METOTREKSAT 1500 mg/m2", headings)
    assert once == "II. LECZENIE > 10. KONSOLIDACJA III\n\nMETOTREKSAT 1500 mg/m2"
    assert with_heading_prefix(once, headings) == once


def test_heading_prefix_kept_when_docling_contextualized():
    text = "II. LECZENIE\n10. KONSOLIDACJA III\nMETOTREKSAT"
    assert with_heading_prefix(text, DL_META["headings"]) == text


def test_heading_prefix_without_headings():
    assert with_heading_prefix("text", []) == "text"


def test_is_excluded():
    assert is_excluded([1, 2], [1, 2, 3, 4])
    assert not is_excluded([4, 5], [1, 2, 3, 4])
    assert not is_excluded([], [1, 2, 3, 4])
    assert not is_excluded([1], [])


def test_normalize_chunk_drops_dl_meta():
    doc = Document(page_content="METOTREKSAT", metadata={"dl_meta": DL_META, "source": "x"})
    out = normalize_chunk(doc, 7)
    assert out.metadata == {
        "pages": [16, 17],
        "page": 16,
        "section": "II. LECZENIE > 10. KONSOLIDACJA III",
        "has_table": True,
        "chunk_index": 7,
        "source": "protokol_palg_all7.pdf",
    }
    assert out.page_content.startswith("II. LECZENIE > 10. KONSOLIDACJA III\n\n")


def test_select_chunks_excludes_toc_only():
    toc = Document(page_content="spis", metadata={"pages": [2, 3]})
    body = Document(page_content="dawka", metadata={"pages": [4, 5]})
    assert select_chunks([toc, body], [1, 2, 3, 4]) == [body]


def test_point_id_deterministic():
    assert point_id("a.pdf", 1) == point_id("a.pdf", 1)
    assert point_id("a.pdf", 1) != point_id("a.pdf", 2)


def test_titan_request_body_has_dimensions_and_normalize():
    client = MagicMock()
    client.invoke_model.return_value = {
        "body": MagicMock(read=lambda: json.dumps({"embedding": [0.1] * 4}).encode())
    }
    embeddings = get_embeddings(Settings(_env_file=None))
    embeddings.client = client
    embeddings.embed_query("dawka")
    body = json.loads(client.invoke_model.call_args.kwargs["body"])
    assert body == {"inputText": "dawka", "dimensions": 1024, "normalize": True}
