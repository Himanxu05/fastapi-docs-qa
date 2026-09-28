"""Settings, read from env vars or .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    max_retries: int = 10
    reasoning_effort: str | None = "low"  # only used for reasoning models (gpt-oss on groq)

    docs_repo: str = "https://github.com/fastapi/fastapi"
    docs_ref: str = "0.141.1"  # pinned so results are reproducible
    docs_site: str = "https://fastapi.tiangolo.com"

    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    data_dir: Path = Path("data")

    chunk_words: int = 220
    candidates: int = 30  # per retriever, before fusion
    rerank_top: int = 20  # how many fused results go to the reranker
    top_k: int = 5  # chunks given to the LLM
    # dense won in eval/retrieval_eval.py once bge-base was used, see README
    search_mode: str = "dense"
    # ask the LLM to reword the question in the docs' vocabulary and search with both
    query_rewrite: bool = True
    # The reranker scores the top chunks for relevance. Below this we answer "not in the
    # docs" without calling the LLM. -3 kept all 60 eval questions and still caught the
    # obviously off-topic ones; near misses (Django, Rails...) are left to the LLM to refuse.
    relevance_gate: bool = True
    min_relevance: float = -3.0

    @property
    def index_dir(self) -> Path:
        # one folder per embedding model so they can be compared side by side
        return self.data_dir / "index" / self.embed_model.split("/")[-1]


@lru_cache
def get_settings() -> Settings:
    return Settings()
