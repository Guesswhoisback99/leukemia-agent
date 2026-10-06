"""Factories for embeddings and the Qdrant vector store, shared by ingest and the chain."""

from langchain_aws import BedrockEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

from app.config import Settings

# Changing the embedding model or dimension means re-creating the collection (re-run ingest).
EMBEDDING_DIM = 1024
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def get_embeddings(settings: Settings) -> BedrockEmbeddings:
    # model_kwargs are merged into the Titan v2 request body.
    return BedrockEmbeddings(
        model_id=settings.embedding_model_id,
        region_name=settings.aws_region,
        model_kwargs={"dimensions": EMBEDDING_DIM, "normalize": True},
    )


def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def get_vector_store(settings: Settings, client: QdrantClient | None = None) -> QdrantVectorStore:
    """Hybrid store: dense Titan + sparse BM25, fused with RRF."""
    return QdrantVectorStore(
        client=client or get_qdrant_client(settings),
        collection_name=settings.collection_name,
        embedding=get_embeddings(settings),
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
    )


def recreate_collection(client: QdrantClient, settings: Settings) -> None:
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={DENSE_VECTOR: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
        # IDF modifier is required for BM25 scoring.
        sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)},
    )
