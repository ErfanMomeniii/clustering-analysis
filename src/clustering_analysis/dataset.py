"""Access to the prepared feature matrices and the held-out label vector.

The preparation pipeline persists its results as plain ``.npy`` arrays under
``data/processed/``:

  ``X_scaled``  the scaled feature matrix (28 anonymised components plus the
                three engineered features)
  ``X_pca``     the PCA projection retaining 95 % of variance
  ``umap_2d``   the two-dimensional UMAP embedding, for plotting only
  ``labels``    the held-out ``Class`` flag, used exclusively to score results

These loaders live here rather than inside one of the pipeline drivers so that
both drivers, the dashboard and the tests share a single definition of "where
the data is" — previously the second driver had to import from the first, which
coupled two otherwise independent pipelines.
"""

from __future__ import annotations

import numpy as np

from .io_utils import processed_path


def load_matrix(name: str) -> np.ndarray:
    """Load a prepared feature matrix by name, e.g. ``"X_pca"``."""
    return np.load(processed_path(f"{name}.npy"))


def load_labels() -> np.ndarray:
    """Load the held-out ``Class`` vector (never an input to any clustering)."""
    return np.load(processed_path("labels.npy"))


def stratified_subsample(y: np.ndarray, n: int, *, seed: int) -> np.ndarray:
    """Class-proportional subsample indices, sorted ascending.

    Proportional (rather than fraud-enriched) sampling keeps the 0.17 % positive
    rate of the real problem, so metrics measured on a subsample stay comparable
    with the full-data run. Every class contributes at least one row.
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(y))
    picked = []
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts, strict=True):
        share = max(1, int(round(n * count / len(y))))
        share = min(share, count)
        idx = np.where(y == cls)[0]
        picked.append(rng.choice(idx, size=share, replace=False))
    return np.sort(np.concatenate(picked))
