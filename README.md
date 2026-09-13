# rag-agent

## 1. What it does
`rag-agent` is a multi-tenant RAG API for document ingestion and tenant-scoped retrieval.

## 2. Architecture
- FastAPI API
- PostgreSQL + pgvector data model (`documents`, `chunks`, `ingestion_jobs`, tenant tables)
- JWT authentication
- Organization/workspace tenancy isolation
- Prometheus metrics + Grafana dashboard config

## 3. Retrieval pipeline
1. Upload document via API
2. Parse + chunk text
3. Generate embeddings
4. Persist metadata/chunks/embeddings
5. Query with dense+sparse hybrid scoring and cache

## 4. Evaluation
- Dataset: `data/eval_datasets/*.yaml`
- Metrics: retrieval precision/recall, hit-rate/MRR, groundedness, citation correctness, hallucination rate, latency, token usage
- Run:
```bash
python scripts/run_evals.py --dataset data/eval_datasets/hr_full.yaml --no-judge --out benchmarks/results/ci_eval.json
```

## 5. Reliability
- Ingestion lifecycle states: `pending`, `processing`, `completed`, `failed`
- Retry with exponential backoff for embedding generation (idempotent only)
- Duplicate ingestion rejection
- File/MIME/size validation

## 6. Security
- JWT auth on protected routes
- Tenant authorization checks on all workspace/document/query actions
- Prompt-injection and guardrail test coverage retained
- Dependency vulnerability scan in CI (`pip-audit`)

## 7. Observability
- Metrics for request, retrieval, LLM, errors, token usage, ingestion failures, cache hits
- Request correlation via `x-request-id`
- Dashboard: `deploy/grafana/rag_agent_dashboard.json`

## 8. Deployment
- Local stack: `docker-compose.yml` (API + pgvector + Prometheus + Grafana)
- IaC baseline: `infra/terraform/main.tf`
- Environment model: dev/staging/prod by `environment` variable and image tag

## 9. Benchmarks
- Method: `python benchmarks/load/run_load.py --concurrency 5,20,50 --requests 300`
- Current benchmark artifacts are committed at:
  - `benchmarks/load/results/report.md`
  - `benchmarks/load/results/report.json`
- Use the generated report as the source of truth for latest P50/P95/P99, RPS, and failure rate values.

## 10. Design decisions
See ADRs in `docs/adr/`:
- Hybrid retrieval vs dense-only
- pgvector choice
- FastAPI and SSE choices
- Chunking and reranking strategy
- Retry policy scope

## 11. Limitations
- Current Terraform is a minimal baseline and should be expanded to provider-specific resources.
- LLM generation is intentionally lightweight by default (`mock`) for reproducible local/CI runs.
- For 10x traffic: move retrieval cache to shared Redis and run DB read replicas.
- For 100x traffic: split ingestion into async workers, add partitioning/sharding, and dedicated ANN infrastructure.
- Out of scope: autonomous agents, image modalities, and non-RAG chatbot features.

## API quick start
```bash
docker compose up --build
```
OpenAPI: http://localhost:8000/docs
