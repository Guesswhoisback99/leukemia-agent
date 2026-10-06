from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import api


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health_ok(client, monkeypatch):
    monkeypatch.setattr(api, "get_qdrant_client", lambda: MagicMock())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_qdrant_down(client, monkeypatch):
    qdrant = MagicMock()
    qdrant.get_collections.side_effect = ConnectionError("down")
    monkeypatch.setattr(api, "get_qdrant_client", lambda: qdrant)
    response = client.get("/health")
    assert response.status_code == 503
