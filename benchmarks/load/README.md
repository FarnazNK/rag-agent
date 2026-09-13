# RAG API load benchmark

This harness measures request throughput and latency distributions for a simulated RAG API workload.

## Run

```bash
python benchmarks/load/run_load.py --concurrency 5,20,50 --requests 300
python benchmarks/load/run_load.py --quick
```

Artifacts are written to:

- `benchmarks/load/results/report.md`
- `benchmarks/load/results/report.json`

## Reported metrics

- requests/second
- P50/P95/P99 latency
- failure rate by scenario

Use these results as a reproducible baseline for CI and README benchmark sections.
