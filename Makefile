.PHONY: test test-fast lint format clean ingest phase1 verify dashboard

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

ingest:
	dvc repro -f -s ingest

phase1:
	dvc repro

verify:
	$(PY) -c "import clustering_analysis; print('package importable:', clustering_analysis.__version__)"
	pytest tests/ -m "not slow" -q

dashboard:
	@echo "dashboard target enabled in Phase 3"
