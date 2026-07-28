"""Dimensionality reduction: PCA (linear) + UMAP (non-linear)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def fit_pca_for_variance(X: np.ndarray, *, target_variance: float):
    pca_full = PCA().fit(X)
    cum = pca_full.explained_variance_ratio_.cumsum()
    n = int(np.searchsorted(cum, target_variance) + 1)
    n = min(n, X.shape[1])
    pca = PCA(n_components=n).fit(X)
    return pca, n


def pca_explained_variance_curve(X: np.ndarray, *, max_components: int) -> np.ndarray:
    k = min(max_components, X.shape[1])
    pca = PCA(n_components=k).fit(X)
    return pca.explained_variance_ratio_.cumsum()


def umap_embed(
    X: np.ndarray, *, n_neighbors: int, min_dist: float, n_components: int, seed: int
) -> np.ndarray:
    import umap  # imported lazily — UMAP is heavy

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(X)
