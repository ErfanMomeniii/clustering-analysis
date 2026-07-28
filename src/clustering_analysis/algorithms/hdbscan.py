"""HDBSCAN (density-based family).

``k`` is ignored — HDBSCAN discovers cluster count from density. Noise points
are labelled -1, matching the convention used by the metrics layer.
"""

from __future__ import annotations

import numpy as np

from .base import register


@register("hdbscan")
def fit_hdbscan(
    X: np.ndarray,
    k: int,  # ignored: density-defined cluster count
    *,
    seed: int = 0,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    **_,
) -> np.ndarray:
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )
    return clusterer.fit_predict(X).astype(int)


def fit_hdbscan_scored(
    X: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
):
    """Fit HDBSCAN and return (labels, n_clusters, persistence_scores)."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(X).astype(int)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    persistence = getattr(clusterer, "cluster_persistence_", None)
    return labels, n_clusters, persistence
