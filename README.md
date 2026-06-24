# Clustering Analysis on DS-10 (Credit Card Fraud)

Final project for **Advanced Data Mining**, Spring 1404–1405, KNTU Faculty of Computer Engineering.
Instructor: Dr. Pishgoo. Head TA: Alireza Ghorbani.

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,pipeline]"
make verify          # 1% smoke test
make phase1          # full Phase 1 pipeline
```

## Layout

- `src/clustering_analysis/` — Python package (source of truth)
- `notebooks/` — Phase notebooks (call into the package, render figures)
- `pipelines/dvc.yaml` — Stage DAG
- `reports/` — Phase reports + analysis plan + AI usage disclosure
- `tests/` — pytest unit + property + integration tests
- `params.yaml` — All seeds and hyperparameters
- `Makefile` — Convenience targets (`make phase1`, `make test`, `make verify`)

## Reproducibility

All seeds are centralised in `params.yaml`. Re-run with `dvc repro` to reproduce
every stage's output from raw data.
