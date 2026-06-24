.PHONY: test test-fast lint format clean

PY := python

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	ruff check src tests

format:
	ruff check --fix src tests
	black src tests

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
