.PHONY: setup format lint typecheck test

setup:
	uv venv
	uv pip install -e ".[dev]"
	pre-commit install || echo "Consider installing pre-commit globally."

format:
	ruff format signbridge tests
	ruff check --select I --fix signbridge tests

lint:
	ruff check signbridge tests

typecheck:
	mypy --strict signbridge

test:
	pytest tests/
