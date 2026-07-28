"""Clustering tendency: Hopkins statistic + VAT ordering (brief §2.7)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def hopkins_statistic(X: np.ndarray, *, sample_size: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    rng_uniform = np.random.default_rng(seed + 1)
    n, d = X.shape
    if sample_size >= n:
        sample_size = max(2, n // 10)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    X_sample = X[sample_idx]
    mins, maxs = X.min(axis=0), X.max(axis=0)
    X_uniform = rng_uniform.uniform(mins, maxs, size=(sample_size, d))
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    w, _ = nbrs.kneighbors(X_sample)
    u, _ = nbrs.kneighbors(X_uniform)
    w_d = (w[:, 1] ** d).sum()
    u_d = (u[:, 0] ** d).sum()
    return float(u_d / (u_d + w_d))


def vat_ordering(X: np.ndarray) -> list[int]:
    """Return a Prim's-MST traversal of indices that surfaces block structure."""
    D = squareform(pdist(X))
    n = D.shape[0]
    in_tree = [False] * n
    start = int(np.argmax(D.sum(axis=1)))
    order = [start]
    in_tree[start] = True
    dist_to_tree = D[start].copy()
    for _ in range(n - 1):
        candidates = np.where([not t for t in in_tree])[0]
        nxt = int(candidates[np.argmin(dist_to_tree[candidates])])
        order.append(nxt)
        in_tree[nxt] = True
        dist_to_tree = np.minimum(dist_to_tree, D[nxt])
    return order
