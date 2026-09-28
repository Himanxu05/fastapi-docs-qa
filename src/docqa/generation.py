"""Build the prompt from retrieved chunks, call the LLM, and keep track of citations."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Settings
from .retrieval import Hit

NOT_FOUND = "I couldn't find this in the FastAPI docs."

SYSTEM = f"""You answer questions about FastAPI using ONLY the numbered documentation excerpts
you are given.

- Cite the excerpts you used with their numbers in square brackets, like [1] or [2][3],
  right after the sentence they support.
- Include a short code example when the excerpts have one that helps.
- If the excerpts don't contain the answer, reply exactly: "{NOT_FOUND}"
  Do not answer from general knowledge.
- Be concise. No preamble."""

CITATION_RE = re.compile(r"\[(\d+)\]")
# gpt-oss sometimes cites in its own format, e.g. 【2†L1-L9】
ALT_CITATION_RE = re.compile(r"【(\d+)†[^】]*】")

REWRITE = """Rewrite the user's question as a short search query for the FastAPI documentation.
Use the terms the docs use (for example "request body", "path operation", "dependency",
"response model", "middleware", "lifespan"). Output only the query, nothing else."""


class ConfigError(RuntimeError):
    pass


def get_llm(s: Settings, **extra) -> BaseChatModel:
    kwargs = {"model": s.llm_model, "temperature": s.llm_temperature, "max_retries": s.max_retries}
    if s.llm_provider == "groq":
        # gpt-oss "thinks" before answering; low effort roughly halves the tokens per question
        if s.reasoning_effort and "gpt-oss" in s.llm_model:
            kwargs["reasoning_effort"] = s.reasoning_effort
        kwargs.update(extra)
        if not os.getenv("GROQ_API_KEY"):
            raise ConfigError("GROQ_API_KEY is not set, add it to .env")
        from langchain_groq import ChatGroq
        return ChatGroq(**kwargs)
    if s.llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigError("OPENAI_API_KEY is not set, add it to .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**{**kwargs, **extra})
    raise ConfigError(f"unknown LLM_PROVIDER '{s.llm_provider}', use groq or openai")


@dataclass
class Source:
    n: int
    page: str
    title: str
    section: str
    url: str
    text: str

    def to_dict(self, preview: int = 300) -> dict:
        return {"n": self.n, "page": self.page, "title": self.title, "section": self.section,
                "url": self.url, "preview": self.text[:preview]}


def make_sources(hits: list[Hit]) -> list[Source]:
    return [Source(n=i, page=h.chunk.page, title=h.chunk.title, section=h.chunk.section,
                   url=h.chunk.url, text=h.chunk.text) for i, h in enumerate(hits, start=1)]


def build_messages(question: str, sources: list[Source]) -> list:
    context = "\n\n".join(f"[{s.n}] {s.section} ({s.url})\n{s.text}" for s in sources)
    return [SystemMessage(SYSTEM),
            HumanMessage(f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}")]


def normalize_citations(answer: str) -> str:
    return ALT_CITATION_RE.sub(r"[\1]", answer)


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    answer = normalize_citations(answer)
    seen: list[int] = []
    for m in CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def is_refusal(answer: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


async def rewrite_query(llm: BaseChatModel, question: str) -> str:
    out = await llm.ainvoke([SystemMessage(REWRITE), HumanMessage(question)])
    text = out.content if isinstance(out.content, str) else str(out.content)
    return text.strip().strip('"').splitlines()[0][:300] if text.strip() else question


async def stream_answer(llm: BaseChatModel, question: str,
                        sources: list[Source]) -> AsyncIterator[str]:
    async for chunk in llm.astream(build_messages(question, sources)):
        if chunk.content:
            yield chunk.content if isinstance(chunk.content, str) else str(chunk.content)
