"""RAG Agent — a LangGraph-based retrieval-augmented agent.

Public surface kept small on purpose: most consumers only need `Agent`.
`Agent` is lazy-imported so lightweight modules (schemas, scorers) can be
used in environments that haven't installed the full LangGraph stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_agent.schemas import AgentState, RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover
    from rag_agent.agent import Agent

__all__ = ["Agent", "AgentState", "RetrievedChunk"]
__version__ = "0.1.0"


def __getattr__(name: str):  # PEP 562 — lazy module attribute access
    if name == "Agent":
        from rag_agent.agent import Agent as _Agent

        return _Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
