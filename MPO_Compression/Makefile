.PHONY: lint format test test-all test-integration test-cov install-dev clean

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -m "not integration" -v

test-all:
	pytest -v

test-integration:
	pytest -m integration -v --run-integration

test-cov:
	pytest -m "not integration" --cov=mpo_modules --cov-report=html --cov-report=term

install-dev:
	pip install -e ".[dev]"
	pre-commit install

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage *.egg-info dist build
