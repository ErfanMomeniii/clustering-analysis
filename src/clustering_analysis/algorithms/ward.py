"""Ward agglomerative hierarchical clustering."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .base import register


@register("ward")
def fit_ward(X: np.ndarray, k: int, *, seed: int = 0, linkage: str = "ward", **_) -> np.ndarray:
    if k < 2:
        raise ValueError("Ward requires k >= 2")
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return model.fit_predict(X).astype(int)
