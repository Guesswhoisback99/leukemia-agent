from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.chain import REFUSAL, build_chain, format_context
from app.config import Settings

DOC_A = Document(
    page_content="METOTREKSAT 1500 mg/m2",
    metadata={"page": 16, "section": "10. KONSOLIDACJA III"},
)
DOC_B = Document(page_content="ASP", metadata={"page": 20, "section": "REINDUKCJA"})


class StubStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, query, k):
        self.calls.append((query, k))
        return self.results[:k]


class ExplodingLLM(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise AssertionError("LLM must not be called")


def test_format_context():
    assert format_context([DOC_A, DOC_B]) == (
        "[s. 16, 10. KONSOLIDACJA III]\nMETOTREKSAT 1500 mg/m2\n\n---\n\n[s. 20, REINDUKCJA]\nASP"
    )


def test_empty_retrieval_refuses_without_llm():
    store = StubStore([])
    chain = build_chain(Settings(_env_file=None), store=store, llm=ExplodingLLM(responses=["x"]))
    result = chain.invoke({"question": "Stolica Australii?", "top_k": 5})
    assert result["answer"] == REFUSAL
    assert result["retrieved"] == []
    assert store.calls == [("Stolica Australii?", 5)]


def test_answer_carries_retrieved_docs():
    store = StubStore([(DOC_A, 0.81), (DOC_B, 0.42)])
    llm = FakeListChatModel(responses=["  1500 mg/m2 [s. 16, 10. KONSOLIDACJA III] "])
    chain = build_chain(Settings(_env_file=None), store=store, llm=llm)
    result = chain.invoke({"question": "Dawka MTX?", "top_k": 1})
    assert result["answer"] == "1500 mg/m2 [s. 16, 10. KONSOLIDACJA III]"
    assert result["retrieved"] == [(DOC_A, 0.81)]


def test_answer_superscripts_become_plain_digits():
    store = StubStore([(DOC_A, 0.81)])
    llm = FakeListChatModel(responses=["1500 mg/m² [s. 16, 10. KONSOLIDACJA III]"])
    chain = build_chain(Settings(_env_file=None), store=store, llm=llm)
    result = chain.invoke({"question": "Dawka MTX?", "top_k": 5})
    assert result["answer"] == "1500 mg/m2 [s. 16, 10. KONSOLIDACJA III]"


class StubReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, documents, query, top_n):
        self.calls.append((documents, query, top_n))
        # Reverse the candidate order with descending relevance.
        order = list(reversed(range(len(documents))))[:top_n]
        return [{"index": i, "relevance_score": 1.0 - n / 10} for n, i in enumerate(order)]


def test_rerank_reorders_and_cuts_to_top_k():
    docs = [Document(page_content=f"c{i}", metadata={"page": i, "section": ""}) for i in range(6)]
    store = StubStore([(d, 0.5) for d in docs])
    reranker = StubReranker()
    settings = Settings(_env_file=None, rerank_enabled=True, rerank_fetch_k=4)
    chain = build_chain(
        settings, store=store, llm=FakeListChatModel(responses=["ok"]), reranker=reranker
    )
    result = chain.invoke({"question": "q", "top_k": 2})
    assert store.calls == [("q", 4)]
    assert reranker.calls == [(["c0", "c1", "c2", "c3"], "q", 2)]
    assert result["retrieved"] == [(docs[3], 1.0), (docs[2], 0.9)]


def test_reranker_not_called_when_disabled():
    store = StubStore([(DOC_A, 0.8)])
    reranker = StubReranker()
    chain = build_chain(
        Settings(_env_file=None),
        store=store,
        llm=FakeListChatModel(responses=["ok"]),
        reranker=reranker,
    )
    chain.invoke({"question": "q", "top_k": 5})
    assert reranker.calls == []
    assert store.calls == [("q", 5)]


def test_reasoning_blocks_are_not_part_of_answer():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    message = AIMessage(
        content=[
            {"type": "reasoning_content", "reasoning_content": {"text": "Let me think..."}},
            {"type": "text", "text": "1500 mg/m2 [s. 16, 10. KONSOLIDACJA III]"},
        ]
    )
    llm = GenericFakeChatModel(messages=iter([message]))
    chain = build_chain(Settings(_env_file=None), store=StubStore([(DOC_A, 0.8)]), llm=llm)
    result = chain.invoke({"question": "Dawka MTX?", "top_k": 5})
    assert result["answer"] == "1500 mg/m2 [s. 16, 10. KONSOLIDACJA III]"
