"""Configuration and settings.

All runtime config flows through `Settings`. We use pydantic-settings so values
can come from env vars or a .env file without the code caring which.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProjectRoot = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Read once, treat as immutable."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    # --- LLM provider ---
    # We default to Anthropic because the JD calls out modern LLM systems and
    # Claude is a strong default. Swap by changing this one field.
    llm_provider: Literal["anthropic", "openai", "mock"] = "mock"
    llm_model: str = "claude-sonnet-4-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Embeddings — kept separate because cheap models often suffice here.
    embedding_provider: Literal["openai"] = "openai"
    embedding_model: str = "text-embedding-3-small"

    # --- Database / tenancy ---
    database_url: str = "postgresql://localhost/rag_agent"
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    embedding_dimension: int = 1536
    max_upload_size_bytes: int = 5_000_000
    auto_create_schema: bool = False

    # --- Retrieval ---
    top_k_dense: int = 8  # vector search candidates
    top_k_sparse: int = 8  # BM25 candidates
    top_k_final: int = 4  # passed to the LLM after fusion
    min_relevance_score: float = 0.2

    # --- Agent loop ---
    max_iterations: int = 3  # cap reflective re-retrieval cycles
    enable_query_rewrite: bool = True
    enable_grading: bool = True

    # --- Serving ---
    # Wall-clock cap on a single agent run. Generous relative to a normal run
    # (route + rewrite + retrieve + grade + generate against a hosted LLM),
    # but bounded: without it a wedged provider call pins a worker slot until
    # the client gives up, and under load that is how a queue collapses.
    request_timeout_seconds: float = 60.0

    # --- Observability ---
    # If LANGSMITH_API_KEY is set in the env, traces will flow there. Otherwise
    # tracing degrades to structured logs — no hard dependency.
    langsmith_project: str = "rag-agent"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Tests can clear the cache to reload."""
    return Settings()
