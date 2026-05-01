"""Observability: structured logs + optional LangSmith tracing.

Design choice: tracing is optional. If `LANGSMITH_API_KEY` is in the env,
LangChain's auto-tracing kicks in. If not, we still get rich structured logs
and the agent runs identically. No hard dependency on a vendor.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

from rag_agent.config import get_settings


def configure_logging() -> None:
    """Idempotent logging setup. Safe to call multiple times."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def maybe_enable_langsmith() -> bool:
    """Turn on LangSmith tracing if the API key is present.

    Returns True if tracing was enabled, False otherwise. We only set
    `LANGCHAIN_TRACING_V2=true` when we have a key — otherwise LangChain
    emits noisy warnings on every call.
    """
    if not os.environ.get("LANGSMITH_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY"):
        return False

    settings = get_settings()
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    return True


def get_logger(name: str) -> Any:
    """Project-wide logger factory. Always use this, never `logging.getLogger`."""
    return structlog.get_logger(name)
