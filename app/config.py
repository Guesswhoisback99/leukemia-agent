from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings loaded from env vars and `.env`.

    AWS credentials are not settings: boto3 reads them from its default chain
    (`AWS_PROFILE`, `~/.aws`).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aws_region: str = "eu-central-1"

    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dim: int = 1024
    llm_model_id: str = "eu.amazon.nova-2-lite-v1:0"

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "palg_all7"

    pdf_path: str = "data/protokol_palg_all7.pdf"
    # "pypdf" is only an experiment baseline (no section headings, no table structure).
    parser: Literal["docling", "pypdf"] = "docling"
    chunks_path: str = "data/chunks.jsonl"
    excluded_pages: list[int] = [1, 2, 3, 4]
    # Tokenizer used only to size chunks (Titan's tokenizer is not public); multilingual for Polish.
    chunk_tokenizer: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_max_tokens: int = 500
    # Docling inference device. "auto" picks MPS on Macs, which breaks layout detection on
    # Intel Macs with torch 2.2 (no headings, no tables), so the default is CPU.
    docling_device: Literal["auto", "cpu", "mps", "cuda"] = "cpu"

    top_k: int = 5
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    rerank_enabled: bool = False
    rerank_model_id: str = "cohere.rerank-v3-5:0"
    rerank_fetch_k: int = 20

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
