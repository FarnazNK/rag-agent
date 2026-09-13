# Observability

- Prometheus scrape config: `deploy/prometheus/prometheus.yml`
- Grafana dashboard JSON: `deploy/grafana/rag_agent_dashboard.json`
- Metrics include request/retrieval/LLM latency, error rate, token usage, cache hits, ingestion failures.

Import dashboard JSON in Grafana and connect the Prometheus data source to reproduce the standard dashboard.
