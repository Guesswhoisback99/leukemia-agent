from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings loaded from env vars and `.env`.

    AWS credentials are not settings: boto3 reads them from its default chain
    (`AWS_PROFILE`, `~/.aws`).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aws_region: str = "eu-central-1"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    llm_model_id: str = "eu.amazon.nova-2-lite-v1:0"

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "palg_all7"

    top_k: int = 5
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
