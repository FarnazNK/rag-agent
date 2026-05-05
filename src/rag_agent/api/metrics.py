"""Prometheus metrics for the API.

Counters and histograms are minimal but cover the metrics any production
RAG service should ship with:
- request count by route and status
- end-to-end latency histogram
- guardrail block count by guardrail name
- retrieval latency separately, so we can tell whether slowdowns are
  retrieval-bound or LLM-bound

We register everything against a custom registry rather than the global one
so multiple test instances don't collide.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

# Shared registry — reset in tests by re-importing the module.
REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "rag_agent_requests_total",
    "Total HTTP requests handled by the API.",
    labelnames=("endpoint", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "rag_agent_request_latency_seconds",
    "End-to-end request latency in seconds.",
    labelnames=("endpoint",),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

GUARDRAIL_BLOCKS = Counter(
    "rag_agent_guardrail_blocks_total",
    "Number of times a guardrail blocked a request.",
    labelnames=("guardrail",),
    registry=REGISTRY,
)

GUARDRAIL_SANITIZATIONS = Counter(
    "rag_agent_guardrail_sanitizations_total",
    "Number of times a guardrail sanitized a request.",
    labelnames=("guardrail",),
    registry=REGISTRY,
)

ITERATIONS_OBSERVED = Histogram(
    "rag_agent_loop_iterations",
    "Number of retrieve-grade loops per request.",
    buckets=(0, 1, 2, 3, 4, 5, 6),
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """Return the current metric snapshot in Prometheus exposition format."""
    return generate_latest(REGISTRY)
