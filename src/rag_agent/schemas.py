"""Schemas for state and message contracts.

The graph state is a Pydantic model rather than a TypedDict. Slightly heavier
at runtime but worth it: validation errors at node boundaries surface bugs
that would otherwise show up as confusing downstream failures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single chunk returned from retrieval."""

    chunk_id: str
    content: str
    source: str
    score: float = Field(description="Fused/normalized score in [0, 1].")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_context_block(self) -> str:
        """Format for injection into an LLM prompt."""
        return f"[source: {self.source} | score: {self.score:.2f}]\n{self.content}"


class GradingResult(BaseModel):
    """Output of the relevance-grading node."""

    is_relevant: bool
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


RouteDecision = Literal["retrieve", "answer_directly", "refuse"]


class AgentState(BaseModel):
    """The state object that flows through every node in the graph.

    `messages` uses LangGraph's `add_messages` reducer so each node can return
    only the new messages it produced — the framework merges them.
    """

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Per-turn working memory
    query: str = ""
    rewritten_query: str | None = None
    route: RouteDecision | None = None
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    grading: GradingResult | None = None
    iterations: int = 0
    final_answer: str | None = None

    # Trace metadata
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}
