"""HTTP API and the chat page."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .config import get_settings
from .generation import get_llm
from .index import Index
from .pipeline import QA
from .retrieval import MODES, Retriever

load_dotenv()
log = logging.getLogger("docqa")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "qa", None) is not None:  # already set up (tests)
        yield
        return
    s = get_settings()
    path = s.index_dir / "heading"
    if not (path / "chunks.json").exists():
        raise RuntimeError(f"no index at {path}, run `docqa ingest` first")
    retriever = Retriever(Index.load(path), s)
    retriever.search("warm up the models", mode="rerank")
    app.state.qa = QA(retriever, get_llm(s), s)
    log.info("loaded %d chunks", len(retriever.index.chunks))
    yield


app = FastAPI(title="FastAPI docs Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health(request: Request):
    qa: QA = request.app.state.qa
    return {"status": "ok", "chunks": len(qa.retriever.index.chunks),
            "docs_version": qa.s.docs_ref, "model": f"{qa.s.llm_provider}/{qa.s.llm_model}"}


@app.get("/search")
def search(request: Request, q: str = Query(min_length=2), mode: str = "rerank", k: int = 5):
    """Retrieval only, handy for debugging what the LLM will see."""
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    res = request.app.state.qa.retriever.search(q, mode=mode, k=min(k, 20))
    return {"timings_ms": res.timings_ms, "hits": [
        {"score": round(h.score, 3), "ranks": h.ranks, "title": h.chunk.title,
         "section": h.chunk.section, "url": h.chunk.url, "preview": h.chunk.text[:300]}
        for h in res.hits]}


@app.post("/ask")
async def ask(body: AskRequest, request: Request):
    try:
        return await request.app.state.qa.ask(body.question.strip())
    except Exception as e:
        # almost always the LLM provider (bad key, rate limit, outage)
        log.exception("answer failed")
        raise HTTPException(502, f"answer failed: {str(e)[:300]}") from e


@app.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request):
    qa: QA = request.app.state.qa

    async def events():
        try:
            async for event, data in qa.stream(body.question.strip()):
                yield {"event": event, "data": json.dumps(data)}
        except Exception as e:
            log.exception("answer failed")
            yield {"event": "error", "data": json.dumps(str(e)[:300])}

    return EventSourceResponse(events())
