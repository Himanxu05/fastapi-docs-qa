import json

import pytest
from fastapi.testclient import TestClient

from docqa import api
from docqa.pipeline import QA

from .conftest import FakeLLM


@pytest.fixture
def client(retriever, settings):
    api.app.state.qa = QA(retriever, FakeLLM("Use CORSMiddleware [1]."), settings)
    with TestClient(api.app) as c:
        yield c
    api.app.state.qa = None


def test_health(client):
    assert client.get("/health").json()["chunks"] == 4


def test_search(client):
    hits = client.get("/search", params={"q": "CORS frontend other origin", "k": 1}).json()["hits"]
    assert hits[0]["url"].endswith("/tutorial/cors/")
    assert client.get("/search", params={"q": "x y", "mode": "magic"}).status_code == 422


def test_ask(client):
    out = client.post("/ask", json={"question": "How do I allow a frontend on another origin?"}).json()
    assert "CORSMiddleware" in out["answer"] and out["cited"] == [1]


def test_llm_failure_is_a_clean_502(client, retriever, settings):
    class Broken:
        async def astream(self, messages):
            raise RuntimeError("Invalid API Key")
            yield  # makes this an async generator

    api.app.state.qa = QA(retriever, Broken(), settings)
    r = client.post("/ask", json={"question": "How do I allow a frontend on another origin?"})
    assert r.status_code == 502 and "Invalid API Key" in r.json()["detail"]


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_stream_is_server_sent_events(client):
    with client.stream("POST", "/ask/stream", json={"question": "allow frontend other origin CORS"}) as r:
        body = "".join(r.iter_text())
    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event: ")]
    assert events[0] == "sources" and events[-1] == "done"
    done = [line for line in body.splitlines() if line.startswith("data: ")][-1]
    assert json.loads(done.split(": ", 1)[1])["cited"] == [1]


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Ask the FastAPI docs" in r.text
