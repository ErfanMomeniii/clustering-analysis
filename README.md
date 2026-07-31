# Clustering analysis of credit card transactions

Final project for Advanced Data Mining (Spring 1404–1405, KNTU Faculty of Computer
Engineering). Instructor: Dr. Pishgoo.

The dataset is 284,807 card transactions, of which 492 are fraud. Each transaction is described
by 28 anonymised numeric features plus its timestamp and amount. We wanted to know whether
fraud shows up as its own cluster, because if it does, you can flag new fraud patterns without
having labelled examples of them first.

## What we found

Fraud does not form its own cluster. We tried four families of clustering algorithm, five
separate methods for picking the number of clusters, and three neural clustering models. None of
them recovered the fraud/legitimate split: the best agreement with the true labels across every
clustering we ran was 0.010, on a scale where 0 means no better than random.

The clustering still turned out to be useful, just not as a classifier. If you score each
transaction by its distance from the centre of its own cluster and review the worst 1%, you
catch 69% of the fraud in that sample (400 transactions reviewed, 46 of the 67 frauds found).
That is roughly 69 times better than reviewing 400 transactions at random, and the score needs
no labels to compute. Details, caveats and the numbers behind all of this are in
`reports/phase3_report.pdf`.

## Setup

```bash
uv venv && source .venv/bin/activate
make install
make verify      # imports the package and runs the fast tests
```

We don't redistribute the transaction data. Download it from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and unzip it to
`data/raw/creditcard.csv`. The first step checks the file's SHA-256, row count and column count
against `params.yaml` and stops if they don't match, so you'll know straight away if you have
the wrong file rather than finding out from odd results later.

## Running it

```bash
make all         # the whole analysis, in order
make dashboard   # browse the results in a Streamlit app
make             # list every command
```

The analysis is a chain of four steps. Each one writes its output to disk, so you can re-run any
step by itself and the ones after it will pick up the new output.

`make prepare-data`
Drops duplicate rows, derives features from the timestamp and amount, puts all features on a
comparable scale, runs PCA and UMAP, and checks whether the data has any cluster structure worth
pursuing. Writes the prepared arrays under `data/`, the cleaning log to
`reports/decision_log.md`, and the first figures.

`make compare-algorithms`
Runs K-Means, Ward, HDBSCAN and a Gaussian mixture, picks the number of clusters using five
criteria, scores every result on four internal and five external measures, and re-runs
everything under different random seeds to see what holds up. Writes
`results/phase2/summary.json` plus the tables and figures the Phase 2 report uses.

`make deep-clustering`
Trains an autoencoder + K-Means, DEC, and a DEC variant with a contrastive loss, and compares
them against the winner from the previous step on identical data. Then extracts decision rules
for each cluster, checks the clusters for bias by amount and time of day, and measures how much
the data shifts between the two days it covers. Writes `results/phase3/summary.json`, more
tables and figures, and a record of the trained model under `models/`.

`make reports`
Rebuilds the four PDFs in `reports/`.

Because the course is organised in phases, `make phase1`, `make phase2` and `make phase3` work
as aliases for the three analysis steps.

### How long it takes

`make compare-algorithms` takes about 40 minutes on a laptop. Almost all of that is HDBSCAN
working through all 283,726 rows: it slows down badly in 28 dimensions, roughly quadratically,
which is one of the things the Phase 2 report discusses. The other steps take a couple of
minutes each, except `make prepare-data`, where UMAP needs around 20 minutes.

If you only want to reword a report, you don't need to re-run anything:

```bash
python -m clustering_analysis.run_portfolio --tables-only
python -m clustering_analysis.run_deep_clustering --tables-only
make reports
```

## Layout

```
src/clustering_analysis/
  run_portfolio.py          entry point for the algorithm comparison
  run_deep_clustering.py    entry point for the neural models and the audits
  dashboard.py              the Streamlit app

  ingest.py, clean.py, features.py, scaling.py, reduce.py   preparing the data
  tendency.py, distance.py, schemas.py                      structure tests, distance metrics, validation
  algorithms/, k_selection.py, metrics.py                   the algorithms, choosing k, scoring
  hierarchical.py, evaluation.py, stability.py              dendrograms, comparison, robustness
  deep/                                                     the three neural models
  interpretation.py, downstream.py, fairness.py             explaining and auditing clusters
  drift.py, registry.py, significance.py                    drift, model records, statistics
  dataset.py, io_utils.py, config.py, seeds.py, reporting.py  shared plumbing

notebooks/   the exploratory notebook, committed with its outputs
reports/     the PDFs, and the tables and figures they are built from
results/     every number each step produced, as JSON
models/       what was trained, when, on which data, with which settings
tests/       the test suite
params.yaml  all settings and random seeds
dvc.yaml     how the steps depend on one another
```

## Reproducibility

The reports don't contain any hand-typed numbers. Each step writes its results to
`results/*/summary.json` and its tables to `reports/tables/*.tex`, and the LaTeX pulls those
files in. So if a re-run changes a result, the report changes with it.

Prose is the weak point in that arrangement, since someone has to write it. There's a test
(`tests/unit/test_report_claims.py`) that reads the saved results and checks that the sentences
quoting them still match, which is what stops a re-run from quietly contradicting the write-up.

Other things worth knowing:

- All random seeds are in `params.yaml`, and each step records the settings it actually ran with
  next to its results.
- Every step validates the shape and types of the data it receives, so a change upstream fails
  loudly.
- The notebook and its plain-Python twin are kept in sync automatically, so what's committed is
  what runs.
- `requirements.txt` pins the versions that produced the committed results. `pyproject.toml` has
  looser ranges for development.

## Tests

```bash
make test        # everything
make test-fast   # skips the slow end-to-end runs
make lint
```
