from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "rag_agent_requests_total",
    "Total API requests.",
    ("endpoint", "status"),
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "rag_agent_request_latency_seconds",
    "Request latency.",
    ("endpoint",),
    registry=REGISTRY,
)
RETRIEVAL_LATENCY = Histogram(
    "rag_agent_retrieval_latency_seconds",
    "Retrieval latency.",
    registry=REGISTRY,
)
LLM_LATENCY = Histogram(
    "rag_agent_llm_latency_seconds",
    "Answer generation latency.",
    ("endpoint",),
    registry=REGISTRY,
)
INGESTION_FAILURES = Counter(
    "rag_agent_ingestion_failures_total",
    "Ingestion failures.",
    registry=REGISTRY,
)
CACHE_HITS = Counter(
    "rag_agent_cache_hits_total",
    "Retrieval cache hits.",
    registry=REGISTRY,
)
TOKEN_USAGE = Counter(
    "rag_agent_token_usage_total",
    "Approximate token usage.",
    ("kind",),
    registry=REGISTRY,
)
ERRORS = Counter(
    "rag_agent_errors_total",
    "Error responses by endpoint.",
    ("endpoint",),
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)
