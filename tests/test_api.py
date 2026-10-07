from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app import api
from app.chain import REFUSAL

DOC = Document(
    page_content="METOTREKSAT " * 50,
    metadata={"page": 16, "section": "10. KONSOLIDACJA III", "chunk_index": 42},
)


@pytest.fixture
def client():
    # No `with`: the lifespan (which builds the real chain) does not run.
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def override_chain(fn):
    api.app.dependency_overrides[api.get_chain] = lambda: RunnableLambda(fn)


def test_health_ok(client, monkeypatch):
    monkeypatch.setattr(api, "get_qdrant_client", lambda _: MagicMock())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_qdrant_down(client, monkeypatch):
    qdrant = MagicMock()
    qdrant.get_collections.side_effect = ConnectionError("down")
    monkeypatch.setattr(api, "get_qdrant_client", lambda _: qdrant)
    response = client.get("/health")
    assert response.status_code == 503


def test_ask_response_shape(client):
    seen = {}

    def fake_chain(inputs):
        seen.update(inputs)
        return {
            "answer": "1500 mg/m2 [s. 16, 10. KONSOLIDACJA III]",
            "retrieved": [(DOC, 0.8)],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    override_chain(fake_chain)
    response = client.post("/ask", json={"question": "Dawka MTX?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("1500 mg/m2")
    assert body["model"] == api.get_settings().llm_model_id
    assert body["sources"] == [
        {
            "section": "10. KONSOLIDACJA III",
            "page": 16,
            "snippet": DOC.page_content[: api.SNIPPET_CHARS],
            "score": 0.8,
        }
    ]
    assert seen == {"question": "Dawka MTX?", "top_k": api.get_settings().top_k}


def test_ask_custom_top_k_and_refusal(client):
    override_chain(lambda inputs: {"answer": REFUSAL, "retrieved": [], "usage": None})
    response = client.post("/ask", json={"question": "Stolica Australii?", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["answer"] == REFUSAL
    assert response.json()["sources"] == []


def ask_log_line(caplog) -> str:
    return next(m for r in caplog.records if (m := r.getMessage()).startswith("ask "))


def test_ask_logs_hits_and_tokens(client, caplog):
    # Shape returned by ChatBedrockConverse: reasoning is counted inside output_tokens.
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    override_chain(
        lambda inputs: {"answer": "1500 mg/m2", "retrieved": [(DOC, 0.8)], "usage": usage}
    )
    with caplog.at_level("INFO", logger="app.api"):
        client.post("/ask", json={"question": "Dawka MTX?"})
    line = ask_log_line(caplog)
    assert "hits=[(42, 16, 0.8)]" in line
    assert "refused=False" in line
    assert "input_tokens=10 output_tokens=5 " in line


def test_ask_logs_refusal_without_tokens(client, caplog):
    override_chain(lambda inputs: {"answer": REFUSAL, "retrieved": [], "usage": None})
    with caplog.at_level("INFO", logger="app.api"):
        client.post("/ask", json={"question": "Stolica Australii?"})
    line = ask_log_line(caplog)
    assert "hits=[] refused=True" in line
    assert "input_tokens=None" in line


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x", "top_k": 50}, {}])
def test_ask_validation(client, payload):
    override_chain(lambda inputs: pytest.fail("chain must not run"))
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_chain_error_returns_502(client):
    def broken(inputs):
        raise RuntimeError("Bedrock throttled")

    override_chain(broken)
    response = client.post("/ask", json={"question": "Dawka MTX?"})
    assert response.status_code == 502
    assert "Bedrock" not in response.text
