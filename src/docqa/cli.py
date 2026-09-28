"""
docqa ingest [--chunking heading|fixed]
docqa ask "How do I upload a file?"
docqa search "upload file" [--mode bm25|dense|hybrid|rerank]
docqa serve [--port 8000]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="docqa")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="download the docs and build the index")
    ing.add_argument("--chunking", default="heading", choices=["heading", "fixed"])
    ask = sub.add_parser("ask", help="answer a question")
    ask.add_argument("question")
    srch = sub.add_parser("search", help="show what retrieval returns")
    srch.add_argument("query")
    srch.add_argument("--mode", default="rerank")
    srch.add_argument("-k", type=int, default=5)
    srv = sub.add_parser("serve", help="run the API + chat page")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("docqa").setLevel(logging.INFO)
    s = get_settings()

    if args.cmd == "ingest":
        from .index import build_index
        path = build_index(s, args.chunking)
        print(f"index written to {path}")
        return 0

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("docqa.api:app", host=args.host, port=args.port)
        return 0

    from .index import Index
    from .retrieval import Retriever
    path = s.index_dir / "heading"
    if not (path / "chunks.json").exists():
        print("no index yet, run `docqa ingest` first", file=sys.stderr)
        return 2
    retriever = Retriever(Index.load(path), s)

    if args.cmd == "search":
        res = retriever.search(args.query, mode=args.mode, k=args.k)
        for i, h in enumerate(res.hits, 1):
            print(f"{i}. [{h.score:.2f}] {h.chunk.section}\n   {h.chunk.url}")
        print(res.timings_ms)
        return 0

    from .generation import ConfigError, get_llm
    from .pipeline import QA
    try:
        qa = QA(retriever, get_llm(s), s)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = asyncio.run(qa.ask(args.question))
    print(out["answer"], "\n")
    for src in out["cited_sources"]:
        print(f"[{src['n']}] {src['section']}\n    {src['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
