"""K-Means (partitioning family)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .base import register


@register("kmeans")
def fit_kmeans(X: np.ndarray, k: int, *, seed: int, n_init: int = 10, **_) -> np.ndarray:
    if k < 2:
        raise ValueError("K-Means requires k >= 2")
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    return km.fit_predict(X).astype(int)
