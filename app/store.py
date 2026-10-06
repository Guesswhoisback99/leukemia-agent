"""Factories for embeddings and the Qdrant vector store, shared by ingest and the chain."""

from typing import Literal

from langchain_aws import BedrockEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

from app.config import Settings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

_RETRIEVAL_MODES = {"dense": RetrievalMode.DENSE, "hybrid": RetrievalMode.HYBRID}


def get_embeddings(settings: Settings) -> BedrockEmbeddings:
    # model_kwargs are merged into the Titan v2 request body.
    return BedrockEmbeddings(
        model_id=settings.embedding_model_id,
        region_name=settings.aws_region,
        model_kwargs={"dimensions": settings.embedding_dim, "normalize": True},
    )


def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name="Qdrant/bm25")


def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def get_vector_store(
    settings: Settings,
    mode: Literal["dense", "hybrid"],
    client: QdrantClient | None = None,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client or get_qdrant_client(settings),
        collection_name=settings.collection_name,
        embedding=get_embeddings(settings),
        sparse_embedding=get_sparse_embeddings() if mode == "hybrid" else None,
        retrieval_mode=_RETRIEVAL_MODES[mode],
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
    )


def recreate_collection(client: QdrantClient, settings: Settings) -> None:
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            DENSE_VECTOR: VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)
        },
        # IDF modifier is required for BM25 scoring.
        sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)},
    )
