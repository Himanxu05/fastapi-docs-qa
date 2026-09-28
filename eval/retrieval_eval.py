"""Compare retrieval setups on eval/questions.jsonl. No LLM needed.

    python eval/retrieval_eval.py
    python eval/retrieval_eval.py --rerankers BAAI/bge-reranker-base --embed-models BAAI/bge-base-en-v1.5

A question counts as a hit@k if any of the top k chunks comes from one of its
gold pages. MRR uses the rank of the first chunk from a gold page (top 10).
Out-of-scope questions are used to tune the relevance gate: the reranker score of
the best dense hit should be low for them and high for real questions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from docqa.config import Settings
from docqa.index import Index, build_index, load_embedder
from docqa.retrieval import Retriever


def load_questions() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open()]


def first_gold_rank(hits, gold: list[str]) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.chunk.page in gold:
            return rank
    return None


def evaluate(retriever: Retriever, mode: str, questions: list[dict]) -> dict:
    ranks, times, by_type = [], [], {}
    for q in questions:
        t = time.perf_counter()
        res = retriever.search(q["question"], mode=mode, k=10)
        times.append((time.perf_counter() - t) * 1000)
        r = first_gold_rank(res.hits, q["gold"])
        ranks.append(r)
        by_type.setdefault(q["type"], []).append(r)

    def hit(rs, k):
        return sum(1 for r in rs if r and r <= k) / len(rs)

    return {
        "hit@1": hit(ranks, 1),
        "hit@5": hit(ranks, 5),
        "mrr@10": sum(1 / r for r in ranks if r) / len(ranks),
        "hit@5 keyword": hit(by_type.get("keyword", [None]), 5),
        "hit@5 paraphrase": hit(by_type.get("paraphrase", [None]), 5),
        "p50 ms": statistics.median(times),
        "misses": [q["id"] for q, r in zip(questions, ranks, strict=True) if not r or r > 5],
    }


def score_separation(retriever: Retriever, in_scope: list[dict], oos: list[dict]) -> dict:
    """How well the relevance gate (reranker score on the top dense hits) separates
    in-scope from out-of-scope questions, and the best threshold for it."""
    def rel(q):
        hits = retriever.search(q["question"], mode="dense", k=retriever.s.top_k).hits
        return retriever.relevance(q["question"], hits)

    ins = [rel(q) for q in in_scope]
    outs = [rel(q) for q in oos]
    best = None
    for th in sorted(set(ins + outs)):
        refused_oos = sum(x < th for x in outs) / len(outs)
        kept_in = sum(x >= th for x in ins) / len(ins)
        cand = (refused_oos + kept_in, th, refused_oos, kept_in)
        if best is None or cand > best:
            best = cand
    _, th, refused, kept = best
    return {"best threshold": round(th, 2), "oos refused": refused, "in-scope kept": kept,
            "min in-scope": round(min(ins), 2), "max oos": round(max(outs), 2),
            "in-scope scores": sorted(round(x, 2) for x in ins)[:5]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-models", nargs="+",
                    default=["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"])
    ap.add_argument("--rerankers", nargs="+", default=["cross-encoder/ms-marco-MiniLM-L-6-v2"])
    ap.add_argument("--chunking", nargs="+", default=["heading", "fixed"])
    args = ap.parse_args()

    questions = load_questions()
    in_scope = [q for q in questions if q["type"] != "out_of_scope"]
    oos = [q for q in questions if q["type"] == "out_of_scope"]

    rows, gate = [], {}
    for model in args.embed_models:
        base = Settings(_env_file=None, embed_model=model)
        embedder = load_embedder(model)
        short_model = model.split("/")[-1]
        for strategy in args.chunking:
            path = base.index_dir / strategy
            if not (path / "chunks.json").exists():
                print(f"building {short_model}/{strategy} index...", flush=True)
                build_index(base, strategy, embedder=embedder)
            index = Index.load(path)
            r = Retriever(index, base, embedder=embedder)
            for mode in ("bm25", "dense", "hybrid"):
                if mode == "bm25" and model != args.embed_models[0]:
                    continue  # bm25 doesn't depend on the embedding model
                rows.append({"embedding": short_model if mode != "bm25" else "-",
                             "chunking": strategy, "setup": mode,
                             **evaluate(r, mode, in_scope)})
                print(f"  {short_model:<20} {strategy:<8} {mode:<8} done", flush=True)
            for name in args.rerankers:
                rr = Retriever(index, base.model_copy(update={"rerank_model": name}),
                               embedder=embedder)
                rr.search("warm up", mode="rerank")
                rows.append({"embedding": short_model, "chunking": strategy,
                             "setup": f"hybrid + {name.split('/')[-1]}",
                             **evaluate(rr, "rerank", in_scope)})
                if strategy == "heading":
                    gate[short_model] = score_separation(rr, in_scope, oos)
                print(f"  {short_model:<20} {strategy:<8} rerank   done", flush=True)

    print("\n| embedding | chunking | setup | hit@1 | hit@5 | MRR@10 | hit@5 keyword "
          "| hit@5 paraphrase | p50 latency |")
    print("|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        print(f"| {x['embedding']} | {x['chunking']} | {x['setup']} | {x['hit@1']:.0%} "
              f"| {x['hit@5']:.0%} | {x['mrr@10']:.2f} | {x['hit@5 keyword']:.0%} "
              f"| {x['hit@5 paraphrase']:.0%} | {x['p50 ms']:.0f} ms |")
    print("\nrelevance gate (reranker score on top dense hits):")
    for model, g in gate.items():
        print(f"  {model}: {g}")

    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    f = out / f"retrieval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"n_questions": len(in_scope), "rows": rows, "gate": gate},
                            indent=2))
    print(f"\nsaved {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
