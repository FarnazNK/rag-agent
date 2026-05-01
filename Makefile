.PHONY: install dev test lint format ingest ask eval eval-ci clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,tracing]"

test:
	pytest -v

test-fast:
	pytest -v tests/test_schemas.py tests/test_retrieval.py tests/test_scorers.py

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

ingest:
	rag-agent ingest --data-dir data/sample_corpus

ask:
	@test -n "$(Q)" || (echo "Usage: make ask Q='your question'"; exit 1)
	rag-agent ask "$(Q)" --trace

eval:
	python scripts/run_evals.py

eval-fast:
	python scripts/run_evals.py --no-judge

eval-ci:
	python scripts/run_evals.py --fail-under 0.9 --out eval_results.json

clean:
	rm -rf data/chroma .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
