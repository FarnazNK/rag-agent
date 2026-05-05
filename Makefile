.PHONY: install dev test test-fast lint format ingest ask eval eval-fast eval-ci \
        corpus benchmark serve docker-build docker-up docker-down clean help

# Default target — list everything.
help:
	@echo "Targets:"
	@echo "  install       Install the package (no dev deps)"
	@echo "  dev           Install with dev + api + tracing extras"
	@echo "  test          Run all unit tests"
	@echo "  test-fast     Run only the no-network tests"
	@echo "  lint          Ruff + mypy"
	@echo "  format        Ruff format + autofix"
	@echo "  corpus        Generate the extended HR corpus"
	@echo "  ingest        Index the sample corpus"
	@echo "  ask Q='...'   Ask the agent a question with --trace"
	@echo "  eval          Run the full eval suite"
	@echo "  eval-fast     Eval suite without the LLM judge"
	@echo "  eval-ci       Eval with --fail-under 0.7 for CI"
	@echo "  benchmark     Run the comparative benchmark across configs"
	@echo "  serve         Run the FastAPI server locally on :8000"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-up     Start the API stack via docker compose"
	@echo "  docker-down   Stop and remove the docker stack"
	@echo "  clean         Wipe build artifacts and caches"

install:
	pip install -e .

dev:
	pip install -e ".[dev,api,tracing]"

test:
	pytest -v

test-fast:
	pytest -v tests/test_schemas.py tests/test_retrieval.py tests/test_scorers.py tests/test_guardrails.py

lint:
	ruff check src tests scripts benchmarks
	mypy src

format:
	ruff format src tests scripts benchmarks
	ruff check --fix src tests scripts benchmarks

corpus:
	python scripts/generate_corpus.py

ingest:
	rag-agent ingest --data-dir data/sample_corpus_extended

ask:
	@test -n "$(Q)" || (echo "Usage: make ask Q='your question'"; exit 1)
	rag-agent ask "$(Q)" --trace --data-dir data/sample_corpus_extended

eval:
	python scripts/run_evals.py --dataset data/eval_datasets/hr_full.yaml

eval-fast:
	python scripts/run_evals.py --dataset data/eval_datasets/hr_full.yaml --no-judge

eval-ci:
	python scripts/run_evals.py --dataset data/eval_datasets/hr_full.yaml \
		--fail-under 0.7 --out benchmarks/results/ci_run.json

benchmark:
	python benchmarks/compare_configs.py

serve:
	uvicorn rag_agent.api:create_app --factory --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t rag-agent:local .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

clean:
	rm -rf data/chroma benchmarks/results .pytest_cache .ruff_cache .mypy_cache \
	       build dist *.egg-info coverage.xml .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
