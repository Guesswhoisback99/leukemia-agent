"""RAG chain: retrieve chunks from Qdrant, format them, prompt the LLM."""

import logging
from typing import Any, TypedDict

from botocore.config import Config
from langchain_aws import BedrockRerank, ChatBedrockConverse
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_qdrant import QdrantVectorStore

from app.config import Settings
from app.store import get_vector_store

logger = logging.getLogger(__name__)

REFUSAL = "Protokół nie zawiera informacji na ten temat."

SYSTEM_PROMPT = f"""Jesteś asystentem odpowiadającym na pytania o protokół leczenia PALG ALL7.

Zasady:
1. Odpowiadaj wyłącznie na podstawie fragmentów protokołu podanych w kontekście. \
Nie korzystaj z żadnej innej wiedzy.
2. Bezpośrednio po każdym stwierdzeniu (w tym samym zdaniu) dodaj źródło dokładnie \
w formacie [s. <strona>, <sekcja>], przepisując etykietę widoczną nad fragmentem, \
np. [s. 16, 10. KONSOLIDACJA III]. Nie twórz osobnej listy źródeł na końcu.
3. Nigdy nie wymyślaj dawek. Nie przenoś dawek ani zaleceń między grupami pacjentów \
(np. wiekowymi lub ryzyka), jeśli kontekst nie mówi tego wprost.
4. Jeśli kontekst nie zawiera odpowiedzi na pytanie, cała Twoja odpowiedź musi brzmieć \
dokładnie: {REFUSAL}
Nie zmieniaj tego zdania, niczego do niego nie dodawaj i nie podawaj źródła.
5. Odpowiadaj po polsku, zwięźle."""

HUMAN_PROMPT = "Kontekst:\n\n{context}\n\nPytanie: {question}"

# The model turns Docling's flattened "mg/m 2" into "mg/m²" whatever the prompt says;
# keep units plain so they match the protocol text.
_SUPERSCRIPTS = str.maketrans("²³", "23")

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])


class ChainResult(TypedDict):
    answer: str
    retrieved: list[tuple[Document, float]]
    usage: dict[str, Any] | None


def retrieve(
    store: QdrantVectorStore,
    question: str,
    top_k: int,
    *,
    reranker: BedrockRerank | None = None,
    fetch_k: int = 20,
) -> list[tuple[Document, float]]:
    """Top-k chunks with scores. With a reranker, rerank `fetch_k` candidates and keep `top_k`;
    the score is then the rerank relevance score."""
    if reranker is None:
        return store.similarity_search_with_score(question, k=top_k)
    candidates = store.similarity_search_with_score(question, k=max(fetch_k, top_k))
    if not candidates:
        return []
    ranked = reranker.rerank([doc.page_content for doc, _ in candidates], question, top_n=top_k)
    return [(candidates[r["index"]][0], r["relevance_score"]) for r in ranked]


def get_reranker(settings: Settings) -> BedrockRerank | None:
    if not settings.rerank_enabled:
        return None
    return BedrockRerank(
        model_arn=(
            f"arn:aws:bedrock:{settings.aws_region}::foundation-model/{settings.rerank_model_id}"
        ),
        region_name=settings.aws_region,
        # Cohere Rerank 3.5 on-demand quota is 3 requests/minute by default: back off, don't fail.
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )


def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[s. {doc.metadata['page']}, {doc.metadata['section']}]\n{doc.page_content}"
        for doc in docs
    )


def get_llm(settings: Settings) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=settings.llm_model_id, region_name=settings.aws_region, temperature=0
    )


def _log_prompt(prompt: PromptValue) -> PromptValue:
    logger.debug("Prompt:\n%s", prompt.to_string())
    return prompt


def _to_result(inputs: dict[str, Any]) -> ChainResult:
    message: AIMessage = inputs["message"]
    return {
        "answer": message.text.strip().translate(_SUPERSCRIPTS),
        "retrieved": inputs["retrieved"],
        "usage": message.usage_metadata,
    }


def build_chain(
    settings: Settings,
    *,
    store: QdrantVectorStore | None = None,
    llm: BaseChatModel | None = None,
    reranker: BedrockRerank | None = None,
) -> Runnable[dict[str, Any], ChainResult]:
    """Chain over `{"question": str, "top_k": int}`. Clients are built once, here."""
    store = store if store is not None else get_vector_store(settings, settings.retrieval_mode)
    llm = llm if llm is not None else get_llm(settings)
    reranker = (reranker or get_reranker(settings)) if settings.rerank_enabled else None

    generate = (
        RunnablePassthrough.assign(context=lambda x: format_context([d for d, _ in x["retrieved"]]))
        | PROMPT
        | RunnableLambda(_log_prompt)
        | llm
    )
    answer = RunnableBranch(
        (lambda x: not x["retrieved"], RunnableLambda(lambda _: AIMessage(content=REFUSAL))),
        generate,
    )
    return (
        RunnablePassthrough.assign(
            retrieved=lambda x: retrieve(
                store,
                x["question"],
                x["top_k"],
                reranker=reranker,
                fetch_k=settings.rerank_fetch_k,
            )
        )
        | RunnablePassthrough.assign(message=answer)
        | RunnableLambda(_to_result)
    )
