import pytest

from docqa.index import tokenize
from docqa.retrieval import rrf


def test_tokenize_splits_identifiers_and_drops_stopwords():
    assert tokenize("How do I use response_model?") == ["use", "response_model", "response", "model"]


def test_rrf_rewards_agreement():
    fused = rrf({"a": [1, 2, 3], "b": [3, 1, 4]})
    order = [idx for idx, _, _ in fused]
    assert order[:2] == [1, 3]  # both lists rank them high
    assert fused[0][2] == {"a": 1, "b": 2}


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "rerank"])
def test_every_mode_finds_the_obvious_page(retriever, mode):
    res = retriever.search("How do I upload a file with UploadFile?", mode=mode, k=2)
    assert res.hits[0].chunk.page == "tutorial/request-files"
    assert "retrieve" in res.timings_ms


def test_rerank_scores_and_ranks(retriever):
    res = retriever.search("run a function after returning a response", mode="rerank", k=2)
    top = res.hits[0]
    assert top.chunk.page == "tutorial/background-tasks"
    assert top.score > res.hits[1].score
    assert "fused" in top.ranks and "rerank" in res.timings_ms


def test_bad_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")
