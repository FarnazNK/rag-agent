.PHONY: dev test lint format typecheck eval benchmark serve docker-up docker-down migrate

dev:
	pip install -e ".[dev,api,tracing]"

test:
	pytest -ra

lint:
	ruff check src tests scripts benchmarks
	ruff format --check src tests scripts benchmarks

typecheck:
	mypy src

format:
	ruff format src tests scripts benchmarks
	ruff check --fix src tests scripts benchmarks

migrate:
	alembic upgrade head

eval:
	python scripts/run_evals.py --dataset data/eval_datasets/hr_full.yaml --no-judge --out benchmarks/results/latest_eval.json

benchmark:
	python benchmarks/load/run_load.py --out benchmarks/load/results/report.md

serve:
	uvicorn rag_agent.api:create_app --factory --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
