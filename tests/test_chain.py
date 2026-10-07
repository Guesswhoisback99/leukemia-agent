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


def test_answer_logged_at_debug(caplog):
    store = StubStore([(DOC_A, 0.81)])
    llm = FakeListChatModel(responses=["1500 mg/m2 [s. 16, 10. KONSOLIDACJA III]"])
    chain = build_chain(Settings(_env_file=None), store=store, llm=llm)
    with caplog.at_level("DEBUG", logger="app.chain"):
        chain.invoke({"question": "Dawka MTX?", "top_k": 5})
    messages = [r.getMessage() for r in caplog.records]
    assert all("\n" not in m for m in messages)
    assert any(m.startswith("Prompt: '") and "METOTREKSAT" in m for m in messages)
    assert "Answer: '1500 mg/m2 [s. 16, 10. KONSOLIDACJA III]'" in messages
