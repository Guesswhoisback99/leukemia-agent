import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.chain import ChainResult, build_chain
from app.config import get_settings
from app.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 300


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    app.state.chain = build_chain(settings)
    yield


app = FastAPI(title="leukemia-agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    section: str
    page: int
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=5)


def get_chain(request: Request) -> Runnable[dict[str, Any], ChainResult]:
    return request.app.state.chain


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_qdrant_client().get_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Qdrant unreachable") from exc
    return {"status": "ok"}


@app.post("/ask")
def ask(
    body: AskRequest,
    chain: Annotated[Runnable[dict[str, Any], ChainResult], Depends(get_chain)],
) -> AskResponse:
    settings = get_settings()
    top_k = body.top_k or settings.top_k
    started = time.perf_counter()
    try:
        result = chain.invoke({"question": body.question, "top_k": top_k})
    except Exception as exc:
        logger.exception("ask failed question=%r", body.question)
        raise HTTPException(status_code=502, detail="Upstream service error") from exc
    latency_ms = (time.perf_counter() - started) * 1000

    # Sources come from the retrieved chunks, never from the LLM output.
    sources = [
        Source(
            section=doc.metadata["section"],
            page=doc.metadata["page"],
            snippet=doc.page_content[:SNIPPET_CHARS],
            score=score,
        )
        for doc, score in result["retrieved"]
    ]
    usage = result["usage"] or {}
    logger.info(
        "ask question=%r hits=%s model=%s input_tokens=%s output_tokens=%s latency_ms=%.0f",
        body.question,
        [(s.page, round(s.score, 3)) for s in sources],
        settings.llm_model_id,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        latency_ms,
    )
    return AskResponse(answer=result["answer"], sources=sources, model=settings.llm_model_id)
