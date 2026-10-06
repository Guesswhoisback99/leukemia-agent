from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient

from app.config import get_settings

app = FastAPI(title="leukemia-agent")


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=5)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_qdrant_client().get_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Qdrant unreachable") from exc
    return {"status": "ok"}
