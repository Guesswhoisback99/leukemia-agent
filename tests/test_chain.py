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
