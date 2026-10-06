from app.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("TOP_K", raising=False)
    settings = Settings(_env_file=None)
    assert settings.aws_region == "eu-central-1"
    assert settings.embedding_model_id == "amazon.titan-embed-text-v2:0"
    assert settings.embedding_dim == 1024
    assert settings.llm_model_id == "eu.amazon.nova-2-lite-v1:0"
    assert settings.collection_name == "palg_all7"
    assert settings.excluded_pages == [1, 2, 3, 4]
    assert settings.top_k == 5
    assert settings.retrieval_mode == "dense"
    assert settings.rerank_enabled is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("EXCLUDED_PAGES", "[1,2]")
    settings = Settings(_env_file=None)
    assert settings.top_k == 8
    assert settings.excluded_pages == [1, 2]


def test_no_aws_key_fields():
    fields = {name.lower() for name in Settings.model_fields}
    assert not any("secret" in f or "access_key" in f for f in fields)
