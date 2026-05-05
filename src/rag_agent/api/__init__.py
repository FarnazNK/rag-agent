"""FastAPI service that exposes the agent over HTTP.

The API is deliberately thin — it adapts HTTP into Agent calls and back. All
business logic stays in the Agent / graph / retrieval modules. That separation
matters because the same Agent should be invokable from a CLI, a Lambda
function, or a batch eval runner without the API layer in the way.
"""

from __future__ import annotations

from rag_agent.api.app import create_app

__all__ = ["create_app"]
