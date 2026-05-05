"""HTTP-facing schemas. Kept separate from the agent's internal schemas so
internal refactors don't break API contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """POST /query and /stream payload."""

    query: str = Field(..., min_length=1, max_length=2000)
    # Reserved for future multi-turn support; not yet plumbed through.
    session_id: str | None = None


class ChunkOut(BaseModel):
    """Retrieved chunk as exposed externally."""

    source: str
    score: float
    snippet: str


class GuardrailEventOut(BaseModel):
    """A single guardrail decision, surfaced to clients for transparency."""

    name: str
    action: Literal["allow", "sanitize", "block"]
    reason: str


class QueryResponse(BaseModel):
    """POST /query 200 body."""

    answer: str
    route: str | None = None
    iterations: int
    chunks: list[ChunkOut]
    latency_ms: float
    run_id: str
    guardrails: list[GuardrailEventOut] = Field(default_factory=list)
    sanitized_query: str | None = None


class HealthResponse(BaseModel):
    """GET /health body."""

    status: Literal["ok", "degraded"]
    corpus_size: int
    vector_count: int
    tracing_enabled: bool


class ErrorResponse(BaseModel):
    """Standard error body — used for guardrail blocks and internal errors."""

    error: str
    detail: str | None = None
    guardrail: str | None = None
