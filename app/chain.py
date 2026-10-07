"""RAG chain: retrieve chunks from Qdrant, format them, prompt the LLM."""

import logging
from typing import Any, TypedDict

from langchain_aws import ChatBedrockConverse
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


def is_refusal(answer: str) -> bool:
    return answer.strip() == REFUSAL


class ChainResult(TypedDict):
    answer: str
    retrieved: list[tuple[Document, float]]
    usage: dict[str, Any] | None


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
    logger.debug("Prompt: %r", prompt.to_string())
    return prompt


def _to_result(inputs: dict[str, Any]) -> ChainResult:
    message: AIMessage = inputs["message"]
    answer = message.text.strip().translate(_SUPERSCRIPTS)
    logger.debug("Answer: %r", answer)
    return {
        "answer": answer,
        "retrieved": inputs["retrieved"],
        "usage": message.usage_metadata,
    }


def build_chain(
    settings: Settings,
    *,
    store: QdrantVectorStore | None = None,
    llm: BaseChatModel | None = None,
) -> Runnable[dict[str, Any], ChainResult]:
    """Chain over `{"question": str, "top_k": int}`. Clients are built once, here."""
    store = store if store is not None else get_vector_store(settings)
    llm = llm if llm is not None else get_llm(settings)

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
            # Scores are RRF fusion scores (hybrid retrieval), not cosine similarities.
            retrieved=lambda x: store.similarity_search_with_score(x["question"], k=x["top_k"])
        )
        | RunnablePassthrough.assign(message=answer)
        | RunnableLambda(_to_result)
    )
