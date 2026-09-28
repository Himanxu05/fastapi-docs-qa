"""Retrieve -> (maybe refuse) -> generate, as a stream of events."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel

from .config import Settings
from .generation import (NOT_FOUND, cited_numbers, is_refusal, make_sources,
                         normalize_citations, rewrite_query, stream_answer)
from .retrieval import Retriever


class QA:
    def __init__(self, retriever: Retriever, llm: BaseChatModel | None, s: Settings,
                 rewrite_llm: BaseChatModel | None = None):
        self.retriever = retriever
        self.llm = llm
        self.s = s
        self.rewrite_llm = rewrite_llm or llm

    async def retrieve(self, question: str):
        """Search, optionally with an LLM-reworded version of the question as well."""
        if self.s.query_rewrite and self.rewrite_llm is not None:
            t = time.perf_counter()
            rewritten = await rewrite_query(self.rewrite_llm, question)
            rewrite_ms = round((time.perf_counter() - t) * 1000, 1)
            res = await asyncio.to_thread(self.retriever.search_many, [question, rewritten])
            res.timings_ms["rewrite"] = rewrite_ms
            return res, rewritten
        return await asyncio.to_thread(self.retriever.search, question, self.s.search_mode), None

    async def stream(self, question: str) -> AsyncIterator[tuple[str, Any]]:
        """Yields ("sources", [...]), then ("token", str)..., then ("done", {...})."""
        # retrieval is CPU-bound (embedding + reranker), keep it off the event loop
        mode = self.s.search_mode
        res, rewritten = await self.retrieve(question)
        timings = dict(res.timings_ms)

        relevance = None
        if self.s.relevance_gate and res.hits:
            t = time.perf_counter()
            # in rerank mode the hits already carry reranker scores
            if mode == "rerank" and rewritten is None:
                relevance = res.top_score  # hits already carry reranker scores
            else:
                phrasings = [question] + ([rewritten] if rewritten else [])
                scores = [await asyncio.to_thread(self.retriever.relevance, q, res.hits)
                          for q in phrasings]
                relevance = max(scores)
            timings["relevance"] = round((time.perf_counter() - t) * 1000, 1)

        if not res.hits or (relevance is not None and relevance < self.s.min_relevance):
            # nothing in the docs looks relevant: skip the LLM, it would only guess
            yield "sources", []
            yield "token", NOT_FOUND
            yield "done", {"refused": True, "reason": "low_relevance", "cited": [],
                           "relevance": relevance, "rewritten": rewritten, "timings_ms": timings}
            return

        sources = make_sources(res.hits)
        yield "sources", [src.to_dict() for src in sources]

        if self.llm is None:
            raise RuntimeError("no LLM configured")
        t = time.perf_counter()
        first_token_ms = None
        parts: list[str] = []
        async for token in stream_answer(self.llm, question, sources):
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - t) * 1000, 1)
            parts.append(token)
            yield "token", token
        answer = "".join(parts)
        timings["first_token"] = first_token_ms
        timings["generate"] = round((time.perf_counter() - t) * 1000, 1)
        refused = is_refusal(answer)
        yield "done", {"refused": refused, "reason": "llm" if refused else None,
                       "cited": cited_numbers(answer, len(sources)), "relevance": relevance,
                       "rewritten": rewritten, "timings_ms": timings}

    async def ask(self, question: str) -> dict:
        out: dict[str, Any] = {"question": question, "answer": ""}
        async for event, data in self.stream(question):
            if event == "sources":
                out["sources"] = data
            elif event == "token":
                out["answer"] += data
            else:
                out.update(data)
        out["answer"] = normalize_citations(out["answer"])
        cited = set(out.get("cited", []))
        out["cited_sources"] = [src for src in out.get("sources", []) if src["n"] in cited]
        return out
