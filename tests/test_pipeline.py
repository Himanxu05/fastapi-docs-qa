from docqa.generation import NOT_FOUND, build_messages, cited_numbers, is_refusal, make_sources
from docqa.pipeline import QA

from .conftest import FakeLLM


def test_cited_numbers_ignores_out_of_range_and_duplicates():
    assert cited_numbers("a [2] b [1][2] c [9]", n_sources=3) == [2, 1]


def test_gpt_oss_citation_format_is_understood():
    from docqa.generation import normalize_citations
    assert cited_numbers("use overrides【2†L1-L9】 and【1†L3】", n_sources=3) == [2, 1]
    assert normalize_citations("x【2†L1-L9】") == "x[2]"


def test_refusal_detection():
    assert is_refusal(NOT_FOUND)
    assert is_refusal("Sorry. " + NOT_FOUND.rstrip("."))
    assert not is_refusal("Use UploadFile [1].")


def test_prompt_numbers_sources(retriever):
    sources = make_sources(retriever.search("upload file", mode="rerank", k=2).hits)
    msgs = build_messages("How?", sources)
    assert "[1] Request Files > UploadFile (https://fastapi.tiangolo.com/tutorial/request-files/)" in msgs[1].content
    assert "Question: How?" in msgs[1].content


async def test_answer_with_citations(retriever, settings):
    llm = FakeLLM("Use UploadFile [1].")
    out = await QA(retriever, llm, settings).ask("How do I upload a file?")
    assert out["answer"].strip() == "Use UploadFile [1]."
    assert out["cited"] == [1] and not out["refused"]
    assert out["cited_sources"][0]["url"].endswith("/tutorial/request-files/")
    assert out["timings_ms"]["first_token"] is not None


async def test_irrelevant_question_skips_the_llm(retriever, settings):
    llm = FakeLLM("should not be called")
    out = await QA(retriever, llm, settings).ask("What is the capital of Australia?")
    assert out["answer"] == NOT_FOUND and out["refused"] and out["reason"] == "low_relevance"
    assert llm.calls == 0


async def test_stream_event_order(retriever, settings):
    events = [e async for e, _ in QA(retriever, FakeLLM("ok [1]"), settings).stream("upload a file")]
    assert events[0] == "sources" and events[-1] == "done" and "token" in events


class RewriteLLM(FakeLLM):
    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage
        self.rewrites = getattr(self, "rewrites", 0) + 1
        return AIMessage(content='"UploadFile receive uploaded file"')


async def test_query_rewrite_searches_with_both_phrasings(retriever, settings):
    cfg = settings.model_copy(update={"query_rewrite": True})
    llm = RewriteLLM("Use UploadFile [1].")
    out = await QA(retriever, llm, cfg).ask("how can users send me a document")
    assert llm.rewrites == 1
    assert out["rewritten"] == "UploadFile receive uploaded file"
    assert out["sources"][0]["page"] == "tutorial/request-files"
    assert "rewrite" in out["timings_ms"]
