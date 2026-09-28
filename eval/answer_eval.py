"""End-to-end answer quality on eval/questions.jsonl. Needs an LLM key.

    python eval/answer_eval.py                  # default setup (dense + query rewrite)
    python eval/answer_eval.py --setups default no-rewrite
    python eval/answer_eval.py --judge          # also check faithfulness with an LLM judge
    python eval/answer_eval.py --limit 20       # quick run on a subset

In-scope questions:
  correct      every `must_include` term appears in the answer
  cited gold   at least one cited source is from a gold page
  refused      said "not in the docs" although it is (bad)
Out-of-scope questions:
  refused      said "not in the docs" (good)
Judge (optional): is every claim in the answer supported by the sources?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from docqa.config import Settings
from docqa.generation import get_llm
from docqa.index import Index, load_embedder
from docqa.pipeline import QA
from docqa.retrieval import Retriever

JUDGE_PROMPT = """You check answers for a documentation assistant.
Given numbered source excerpts and an answer, decide whether every factual claim and code
detail in the answer is supported by the sources. Ignore style. An answer saying the
information isn't in the docs counts as supported."""


class Verdict(BaseModel):
    supported: bool = Field(description="True if every claim is supported by the sources")
    unsupported_claims: list[str] = Field(default_factory=list)


async def judge(llm, question: str, sources: list[dict], answer: str) -> Verdict:
    ctx = "\n\n".join(f"[{s['n']}] {s['preview']}" for s in sources)
    return await llm.with_structured_output(Verdict).ainvoke([
        SystemMessage(JUDGE_PROMPT),
        HumanMessage(f"Sources:\n{ctx}\n\nQuestion: {question}\n\nAnswer:\n{answer}"),
    ])


# overrides on top of the normal settings
SETUPS = {
    "default": {},
    "no-rewrite": {"query_rewrite": False},
    "no-gate": {"relevance_gate": False},
}


async def run(mode: str, questions: list[dict], s: Settings, retriever: Retriever,
              use_judge: bool) -> dict:
    cfg = s.model_copy(update=SETUPS[mode])
    rewrite_llm = get_llm(cfg, reasoning_effort="low") if cfg.llm_provider == "groq" else None
    qa = QA(retriever, get_llm(cfg), cfg, rewrite_llm=rewrite_llm)
    judge_llm = get_llm(cfg) if use_judge else None
    rows = []
    for q in questions:
        t = time.perf_counter()
        try:
            out = await qa.ask(q["question"])
        except Exception as e:
            rows.append({"id": q["id"], "type": q["type"], "error": str(e)[:300]})
            print(f"  [{mode}] {q['id']} ERROR {str(e)[:120]}", flush=True)
            continue
        cited_pages = {src["page"] for src in out["cited_sources"]}
        row = {
            "id": q["id"], "type": q["type"], "answer": out["answer"], "refused": out["refused"],
            "reason": out.get("reason"), "seconds": round(time.perf_counter() - t, 2),
            "first_token_ms": out["timings_ms"].get("first_token"),
            "correct": all(term in out["answer"] for term in q["must_include"]),
            "cited_gold": bool(cited_pages & set(q["gold"])),
        }
        if judge_llm and not out["refused"]:
            try:
                v = await judge(judge_llm, q["question"], out.get("sources", []), out["answer"])
                row["faithful"] = v.supported
                row["unsupported"] = v.unsupported_claims
            except Exception as e:
                row["judge_error"] = str(e)[:200]
        rows.append(row)
        if q["type"] == "out_of_scope":
            status = "refused" if row["refused"] else "ANSWERED (should refuse)"
        else:
            status = f"correct={row['correct']!s:<5} cited_gold={row['cited_gold']!s:<5}" + (
                " REFUSED" if row["refused"] else "")
        print(f"  [{mode}] {q['id']:<6} {status} {row['seconds']}s", flush=True)
    return {"mode": mode, "rows": rows, "summary": summarize(rows)}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    ins = [r for r in ok if r["type"] != "out_of_scope"]
    oos = [r for r in ok if r["type"] == "out_of_scope"]
    judged = [r for r in ins if "faithful" in r]

    def pct(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "answered correctly": pct([r["correct"] for r in ins]),
        "cited a gold page": pct([r["cited_gold"] for r in ins]),
        "wrongly refused": pct([r["refused"] for r in ins]),
        "out-of-scope refused": pct([r["refused"] for r in oos]),
        "refused without LLM call": sum(r["reason"] == "low_relevance" for r in oos),
        "faithful (judge)": pct([r["faithful"] for r in judged]),
        "median first token ms": statistics.median(
            [r["first_token_ms"] for r in ins if r["first_token_ms"]] or [0]),
        "errors": len(rows) - len(ok),
        "n in-scope": len(ins), "n out-of-scope": len(oos),
    }


async def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", nargs="+", default=["default"], choices=list(SETUPS))
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N in-scope + all out-of-scope")
    args = ap.parse_args()

    questions = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open()]
    if args.limit:
        ins = [q for q in questions if q["type"] != "out_of_scope"][:args.limit]
        questions = ins + [q for q in questions if q["type"] == "out_of_scope"]

    s = Settings()
    retriever = Retriever(Index.load(s.index_dir / "heading"), s,
                          embedder=load_embedder(s.embed_model))
    print(f"model {s.llm_provider}/{s.llm_model}, reranker {s.rerank_model}, "
          f"{len(questions)} questions\n")
    results = [await run(m, questions, s, retriever, args.judge) for m in args.setups]

    print("\n| setup | answered correctly | cited a gold page | wrongly refused "
          "| out-of-scope refused | faithful (judge) | median first token |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        x = r["summary"]

        def f(v):
            return "-" if v is None else f"{v:.0%}"
        print(f"| {r['mode']} | {f(x['answered correctly'])} | {f(x['cited a gold page'])} "
              f"| {f(x['wrongly refused'])} | {f(x['out-of-scope refused'])} "
              f"| {f(x['faithful (judge)'])} | {x['median first token ms']:.0f} ms |")

    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    path = out / f"answers-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"model": f"{s.llm_provider}/{s.llm_model}",
                                "results": results}, indent=2))
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
