# fastapi-docs-qa

Ask questions about FastAPI and get answers from the official documentation, with links to the exact
sections they came from. If the docs don't cover it, it says so instead of making something up.

It's a retrieval-augmented generation (RAG) setup, but most of the work went into measuring which parts
actually help. Several "standard" pieces (BM25 hybrid search, a cross-encoder reranker) turned out not to
improve results once the embeddings were good, so they aren't in the default pipeline. See
[Evaluation](#evaluation).

## How it works

```
question
   |
   +--> LLM rewrites it in the docs' vocabulary ("JSON payload" -> "request body ...")
   |
   v
dense search (bge-base embeddings, FAISS) for both phrasings, merged with RRF
   |
   v
relevance check (cross-encoder score of the top chunks)
   |-- too low --> "I couldn't find this in the FastAPI docs."   (no LLM call)
   v
LLM answers from the top 5 chunks only, citing them as [1], [2]...  (streamed)
```

Ingestion:
- Downloads the docs for a pinned FastAPI release (0.141.1).
- Resolves the `{* ../../docs_src/... *}` includes, so the actual code examples end up in the index. Without
  this almost no code gets indexed.
- Splits pages by heading. Each chunk keeps its heading path ("Request Body > Create your data model") and a
  link to that section.

Everything except the LLM runs locally: sentence-transformers on PyTorch (CPU) for embeddings and the
cross-encoder, FAISS for vector search.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, much smaller
pip install -e ".[dev]"
cp .env.example .env        # add GROQ_API_KEY (free at console.groq.com)
docqa ingest                # downloads the docs + models and builds the index (~5 min on CPU)
```

## Usage

```bash
docqa ask "How do I run code after the response is sent?"
docqa search "upload file" --mode dense      # just retrieval, no LLM
docqa serve                                  # API + chat page on http://localhost:8000
```

API:

| endpoint | |
|---|---|
| `POST /ask` | `{"question": "..."}`, returns the answer, the sources it cited, and timings |
| `POST /ask/stream` | the same answer as server-sent events: `sources`, then `token`s, then `done` |
| `GET /search?q=...&mode=dense` | retrieval only, for debugging |
| `GET /health` | |

Docker (the docs, models and index are built into the image):

```bash
docker build -t docs-qa .
docker run -p 8000:8000 --env-file .env docs-qa
```

Don't put quotes around values in `.env` when using `--env-file`: Docker passes them through as part of
the value, so `GROQ_API_KEY="gsk_..."` ends up as an invalid key.

## Evaluation

`eval/questions.jsonl` has 70 hand-written questions:
- 60 answerable ones, each labelled with the page(s) that answer it and a term a correct answer must
  contain. Half use the docs' own wording ("How do I upload a file?"), half are phrased the way people
  actually ask ("How can I make sure the password field never shows up in what my API sends back?").
- 10 that the docs can't answer (Django, Rails, the capital of Australia...).

A script checks every label against the indexed pages, so the gold pages and terms are really there.

### Retrieval

`python eval/retrieval_eval.py`. A question counts as a hit if a chunk from one of its gold pages is in the top k.

| embedding | chunking | setup | hit@1 | hit@5 | MRR@10 | hit@5 paraphrased | p50 latency |
|---|---|---|---|---|---|---|---|
| - | heading | BM25 | 65% | 85% | 0.73 | 87% | 2 ms |
| bge-small | heading | dense | 75% | 93% | 0.82 | 90% | 26 ms |
| bge-small | heading | hybrid (BM25 + dense, RRF) | 72% | 93% | 0.82 | 90% | 28 ms |
| bge-small | heading | hybrid + MiniLM reranker | 75% | 93% | 0.84 | 90% | 1281 ms |
| bge-base | fixed 1000 chars | dense | 85% | 97% | 0.90 | 94% | 58 ms |
| bge-base | heading | hybrid | 75% | 97% | 0.85 | 97% | 53 ms |
| bge-base | heading | hybrid + MiniLM reranker | 77% | 95% | 0.84 | 90% | 1183 ms |
| **bge-base** | **heading** | **dense** | **83%** | **97%** | **0.89** | **94%** | **55 ms** |
| bge-base | heading | dense + query rewrite | 82% | 100% | 0.89 | 100% | +1 LLM call |

What this showed:
- The embedding model mattered most: bge-small to bge-base took hit@1 from 75% to 83%.
- Adding BM25 didn't help on average and made the top result worse with bge-base.
- The cross-encoder reranker added about a second on CPU and, with bge-base, made the ranking slightly
  worse. I also tried `bge-reranker-base`: worse again, and about 8 s per query on CPU.
- Heading-based and fixed-size chunks scored about the same. I kept heading chunks because each one maps to
  a section, so citations link to the exact spot on the page.
- Query rewriting fixed the paraphrased questions dense search still missed (e.g. "accept a JSON payload"
  went from not found to rank 3), so it's on by default. It costs one short LLM call.

The reranker is still used, but only to decide whether anything relevant was found at all. Its score on the
top chunks is a good relevance signal: at the chosen threshold none of the 60 answerable questions get
blocked, and the obviously off-topic ones are refused without calling the LLM. Near misses like "How do I
write Django REST Framework serializers?" do get through, and the LLM is expected to refuse those because
it may only answer from the retrieved docs.

Caveat: 60 questions is small. One question is about 1.7 points, so differences of a few points are noise.

### Answers

`python eval/answer_eval.py` (needs an API key; `--judge` adds an LLM faithfulness check). Per question it
checks:
- whether the answer contains the expected term
- whether it cited a chunk from a gold page
- whether it refused when it shouldn't have, or answered when it should have refused

Full run, all 70 questions, `openai/gpt-oss-20b` on Groq (free tier), default setup:

| scored | answered correctly | cited a gold page | wrongly refused | out-of-scope refused | refused before the LLM |
|---|---|---|---|---|---|
| 70/70 | 55/60 (92%) | 54/60 (90%) | 1/60 | 10/10 | 5/10 |

I went through every failure by hand, because a keyword check can mark a good answer wrong:

- **Real mistakes (3):**
  - q02 suggested middleware instead of `BackgroundTasks` for running code after the response.
  - q36 refused a question the docs do answer (changing the docs page title).
  - q59 gave generic IDE debugger steps and missed the `uvicorn.run()` part the docs describe.
- **Wrongly marked wrong (2):** q56 and q58 were correct. The check wanted "Body" and "template", and the
  answers said "body parameter" and "Template".
- **Correct but uncited (4):** q05, q20, q26 and q40 answered correctly but cited nothing, so they count as
  missing a gold page.

So by hand it's 57/60 correct. The most common problem is missing citations, not wrong answers. After this
run I made the prompt require at least one citation per answer; that change hasn't been measured yet.

The 5 out-of-scope questions that got past the relevance check (Django, React, PyTorch, Spring Boot,
PostgreSQL) were all refused by the LLM.

Time to first token was about 10 s (median), mostly spent waiting on the free tier's 8k tokens/minute limit
on this model.

Earlier, a partial run with `gpt-oss-120b` stopped at Groq's daily token limit after 25 questions (24/25
correct). It found two bugs, both fixed before the full run:
- gpt-oss sometimes cites as `【2†L1-L9】` instead of `[2]`; both formats are now understood.
- The eval script's percentages ignored failed calls. It now saves progress after every question, stops at
  the daily limit, and resumes on the next run.

The `no-rewrite` comparison is still to run. From the partial run, the relevance threshold is too strict
when rewriting is off: it blocked three valid questions. The two settings have to be tuned together.

## Tests

```bash
pytest
ruff check src tests eval
```

The tests use tiny fake embedding/reranker models and a fake LLM, so they run in about a second, with no
downloads and no API key. They cover markdown cleanup, chunking, fusion, the relevance gate, query rewriting,
streaming and the API.

## Layout

```
src/docqa/
  sources.py      download the docs, page URLs
  markdown.py     resolve code includes, admonitions, tabs
  chunking.py     heading-based and fixed-size chunking
  index.py        embeddings + FAISS + BM25
  retrieval.py    bm25 / dense / hybrid / rerank, multi-query fusion, relevance score
  generation.py   prompt, citations, query rewriting
  pipeline.py     retrieve -> relevance check -> generate, as a stream of events
  api.py          FastAPI app
  static/         chat page
  cli.py
eval/             questions + evaluation scripts
tests/
```

## Limitations / next steps

- The eval set is small and written by one person. Growing it with real user questions would make the
  numbers more trustworthy.
- The relevance threshold was picked on the same questions it's evaluated on.
- Only English docs, one FastAPI version. Re-indexing a new release is `DOCS_REF=... docqa ingest`.
- No conversation memory: each question is answered on its own.
