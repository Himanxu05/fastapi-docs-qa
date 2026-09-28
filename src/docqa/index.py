"""Build and load the search index (FAISS for vectors, BM25 for keywords)."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk, chunk_by_heading, chunk_fixed
from .config import Settings
from .sources import fetch_docs, load_pages

log = logging.getLogger(__name__)

# bge models are trained with this prefix on queries (not on documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

STOPWORDS = set("""a an and are as at be by can do does for from how i if in is it of on or
that the this to what when where which with you your my me we our""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words; identifiers like response_model also get split into their parts."""
    tokens = []
    for tok in re.findall(r"[a-z0-9_]+", text.lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(p for p in tok.split("_") if p and p not in STOPWORDS)
    return tokens


def load_embedder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


def embed_query(model, query: str, model_name: str) -> np.ndarray:
    prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
    return model.encode([prefix + query], normalize_embeddings=True,
                        show_progress_bar=False).astype("float32")


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: faiss.Index
    bm25: BM25Okapi
    name: str

    @classmethod
    def load(cls, path: Path) -> Index:
        meta = json.loads((path / "chunks.json").read_text())
        chunks = [Chunk(**c) for c in meta["chunks"]]
        vectors = faiss.read_index(str(path / "vectors.faiss"))
        bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])
        return cls(chunks=chunks, vectors=vectors, bm25=bm25, name=path.name)


def build_index(s: Settings, strategy: str = "heading", embedder=None) -> Path:
    """Fetch docs, chunk, embed, save to data/index/<strategy>/."""
    root = fetch_docs(s.docs_repo, s.docs_ref, s.data_dir / "fastapi")
    pages = load_pages(root, s.docs_site)
    if strategy == "heading":
        chunks = [c for p in pages for c in chunk_by_heading(p, max_words=s.chunk_words)]
    elif strategy == "fixed":
        chunks = [c for p in pages for c in chunk_fixed(p)]
    else:
        raise ValueError(f"unknown chunking strategy {strategy!r}")
    log.info("%d pages -> %d chunks (%s)", len(pages), len(chunks), strategy)

    model = embedder or load_embedder(s.embed_model)
    t = time.perf_counter()
    emb = model.encode([c.embed_text for c in chunks], batch_size=32, normalize_embeddings=True,
                       show_progress_bar=log.isEnabledFor(logging.INFO)).astype("float32")
    log.info("embedded in %.1fs", time.perf_counter() - t)

    # vectors are normalized, so inner product == cosine similarity
    vectors = faiss.IndexFlatIP(emb.shape[1])
    vectors.add(emb)

    out = s.index_dir / strategy
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(vectors, str(out / "vectors.faiss"))
    (out / "chunks.json").write_text(json.dumps({
        "docs_ref": s.docs_ref, "embed_model": s.embed_model, "strategy": strategy,
        "pages": len(pages), "chunks": [c.to_dict() for c in chunks],
    }))
    return out
