"""Does rewriting the question with an LLM help retrieval?

    python eval/rewrite_eval.py

Rewrites are cached in eval/results/rewrites-<model>.json so re-runs don't cost
tokens. Compares dense search on the original question against dense search on
original + rewritten, fused with RRF (what the pipeline does with QUERY_REWRITE=true).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from retrieval_eval import first_gold_rank, load_questions

from docqa.config import Settings
from docqa.generation import get_llm, rewrite_query
from docqa.index import Index
from docqa.retrieval import Retriever


async def get_rewrites(questions: list[dict], s: Settings) -> dict[str, str]:
    cache = ROOT / "eval" / "results" / f"rewrites-{s.llm_model.replace('/', '_')}.json"
    cache.parent.mkdir(exist_ok=True)
    done = json.loads(cache.read_text()) if cache.exists() else {}
    llm = get_llm(s, reasoning_effort="low") if s.llm_provider == "groq" else get_llm(s)
    for q in questions:
        if q["id"] not in done:
            done[q["id"]] = await rewrite_query(llm, q["question"])
            cache.write_text(json.dumps(done, indent=2))
            print(f"  {q['id']}: {q['question'][:50]!r} -> {done[q['id']]!r}", flush=True)
    return done


def report(name: str, ranks: list[int | None]) -> None:
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1) / n
    hit5 = sum(1 for r in ranks if r and r <= 5) / n
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"| {name} | {hit1:.0%} | {hit5:.0%} | {mrr:.2f} |")


async def main() -> None:
    load_dotenv(ROOT / ".env")
    s = Settings()
    questions = [q for q in load_questions() if q["type"] != "out_of_scope"]
    rewrites = await get_rewrites(questions, s)
    r = Retriever(Index.load(s.index_dir / "heading"), s)

    by_type: dict[str, dict[str, list]] = {}
    for q in questions:
        plain = first_gold_rank(r.search(q["question"], mode="dense", k=10).hits, q["gold"])
        both = first_gold_rank(r.search_many([q["question"], rewrites[q["id"]]], k=10).hits,
                               q["gold"])
        for t in ("all", q["type"]):
            by_type.setdefault(t, {"plain": [], "both": []})
            by_type[t]["plain"].append(plain)
            by_type[t]["both"].append(both)
        if plain != both:
            print(f"  {q['id']} rank {plain} -> {both}   ({rewrites[q['id']]!r})")

    print(f"\nembedding {s.embed_model}, rewrites by {s.llm_model}\n")
    print("| setup | hit@1 | hit@5 | MRR@10 |\n|---|---|---|---|")
    for t, d in by_type.items():
        report(f"dense ({t})", d["plain"])
        report(f"dense + rewrite ({t})", d["both"])


if __name__ == "__main__":
    asyncio.run(main())
