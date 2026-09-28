"""Small fakes so tests run without downloading models or calling an LLM."""

from __future__ import annotations

import hashlib
import re

import faiss
import numpy as np
import pytest
from langchain_core.messages import AIMessageChunk
from rank_bm25 import BM25Okapi

from docqa.chunking import Chunk
from docqa.config import Settings
from docqa.index import Index, tokenize

DIM = 64


class FakeEmbedder:
    """Bag of hashed words. Similar wording -> similar vectors, which is enough for tests."""

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), DIM), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeReranker:
    """Score = number of (non-stopword) query words that appear in the passage."""

    def predict(self, pairs, **kwargs):
        return np.array([float(len(set(tokenize(q)) & set(tokenize(p)))) for q, p in pairs])


class FakeLLM:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    async def astream(self, messages):
        self.calls += 1
        for word in self.answer.split(" "):
            yield AIMessageChunk(content=word + " ")


CHUNKS = [
    ("tutorial/body", "Request Body", "Request Body > Create your data model",
     "Declare a request body with a Pydantic BaseModel. FastAPI reads the JSON payload."),
    ("tutorial/background-tasks", "Background Tasks", "Background Tasks",
     "Use BackgroundTasks to run a function after returning a response, like sending an email."),
    ("tutorial/cors", "CORS", "CORS > Use CORSMiddleware",
     "Add CORSMiddleware to allow a frontend on another origin to call the API."),
    ("tutorial/request-files", "Request Files", "Request Files > UploadFile",
     "Use UploadFile to receive an uploaded file."),
]


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, embed_model="fake", candidates=4, rerank_top=4, top_k=2,
                    min_relevance=1.0, query_rewrite=False)


@pytest.fixture
def index() -> Index:
    chunks = [Chunk(id=f"{p}#0", page=p, url=f"https://fastapi.tiangolo.com/{p}/", title=t,
                    section=sec, text=txt) for p, t, sec, txt in CHUNKS]
    emb = FakeEmbedder().encode([c.embed_text for c in chunks])
    vectors = faiss.IndexFlatIP(DIM)
    vectors.add(emb)
    return Index(chunks=chunks, vectors=vectors,
                 bm25=BM25Okapi([tokenize(c.embed_text) for c in chunks]), name="test")


@pytest.fixture
def retriever(index, settings):
    from docqa.retrieval import Retriever
    return Retriever(index, settings, embedder=FakeEmbedder(), reranker=FakeReranker())
