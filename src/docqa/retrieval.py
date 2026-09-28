"""Retrieval: BM25, dense, hybrid (reciprocal rank fusion) and cross-encoder reranking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .config import Settings
from .index import Index, embed_query, load_embedder, tokenize

MODES = ("bm25", "dense", "hybrid", "rerank")
RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # rank in each retriever, for debugging


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]

    @property
    def top_score(self) -> float | None:
        return self.hits[0].score if self.hits else None


def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, dict]]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank).

    Uses ranks instead of raw scores because BM25 scores and cosine similarities
    aren't on the same scale.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, idx in enumerate(ids, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(idx, {})[name] = rank
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(idx, score, ranks[idx]) for idx, score in fused]


class Retriever:
    def __init__(self, index: Index, s: Settings, embedder=None, reranker=None):
        self.index = index
        self.s = s
        self.embedder = embedder or load_embedder(s.embed_model)
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.s.rerank_model, device="cpu")
        return self._reranker

    def _bm25(self, query: str, n: int) -> list[int]:
        scores = self.index.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:n]
        return [int(i) for i in top if scores[i] > 0]

    def _dense(self, query: str, n: int) -> list[int]:
        _, ids = self.index.vectors.search(embed_query(self.embedder, query, self.s.embed_model), n)
        return [int(i) for i in ids[0] if i >= 0]

    def search(self, query: str, mode: str = "rerank", k: int | None = None) -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = k or self.s.top_k
        n = self.s.candidates
        timings: dict[str, float] = {}
        chunks = self.index.chunks

        t = time.perf_counter()
        if mode == "bm25":
            hits = [Hit(chunks[i], 0.0, {"bm25": r}) for r, i in enumerate(self._bm25(query, k), 1)]
            timings["retrieve"] = _ms(t)
            return SearchResult(hits, timings)
        if mode == "dense":
            hits = [Hit(chunks[i], 0.0, {"dense": r})
                    for r, i in enumerate(self._dense(query, k), 1)]
            timings["retrieve"] = _ms(t)
            return SearchResult(hits, timings)

        fused = rrf({"bm25": self._bm25(query, n), "dense": self._dense(query, n)})
        timings["retrieve"] = _ms(t)
        if mode == "hybrid":
            return SearchResult([Hit(chunks[i], sc, rk) for i, sc, rk in fused[:k]], timings)

        t = time.perf_counter()
        pool = fused[:self.s.rerank_top]
        pairs = [(query, chunks[i].embed_text) for i, _, _ in pool]
        scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False) \
            if pairs else []
        hits = [Hit(chunks[i], float(sc), {**rk, "fused": pos})
                for pos, ((i, _, rk), sc) in enumerate(zip(pool, scores, strict=True), start=1)]
        hits.sort(key=lambda h: h.score, reverse=True)
        timings["rerank"] = _ms(t)
        return SearchResult(hits[:k], timings)


    def search_many(self, queries: list[str], k: int | None = None) -> SearchResult:
        """Dense search for several phrasings of the same question, fused with RRF."""
        k = k or self.s.top_k
        t = time.perf_counter()
        fused = rrf({f"q{i}": self._dense(q, self.s.candidates) for i, q in enumerate(queries)})
        hits = [Hit(self.index.chunks[i], sc, rk) for i, sc, rk in fused[:k]]
        return SearchResult(hits, {"retrieve": _ms(t)})

    def relevance(self, query: str, hits: list[Hit]) -> float | None:
        """Best cross-encoder score among the hits: how well the docs match the question."""
        if not hits:
            return None
        pairs = [(query, h.chunk.embed_text) for h in hits]
        return float(max(self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
