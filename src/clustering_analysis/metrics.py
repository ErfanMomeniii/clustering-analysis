"""Internal, external, and significance metrics for clustering evaluation (Phase 2).

Internal metrics score cluster geometry without ground truth. External metrics
compare assignments to the held-out `Class` label. The metric registry makes
the Phase 2 evaluation loop config-driven via `params.yaml::metrics`.

Feasibility note: on DS-10 (283,726 rows) no metric may materialize an n×n
matrix (~643 GB). The Dunn index and any consensus routine therefore operate
on cluster centroids / subsamples, never the full pairwise distance matrix.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import pairwise_distances


def _require_nontrivial_labels(labels: np.ndarray) -> None:
    """Reject label vectors with < 2 distinct clusters or all noise.

    Silhouette / DB / CH are undefined for a single cluster, and HDBSCAN can
    return a degenerate all-noise (-1) result on poorly-separated data.
    """
    unique = np.unique(labels)
    real = unique[unique != -1]
    if len(real) < 2:
        raise ValueError(f"Need >= 2 non-noise clusters for internal metrics, got {len(real)}")


def silhouette(
    X: np.ndarray, labels: np.ndarray, *, sample_size: int | None = None, seed: int = 0
) -> float:
    _require_nontrivial_labels(labels)
    mask = labels != -1
    if sample_size and mask.sum() > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.where(mask)[0], size=sample_size, replace=False)
        return float(silhouette_score(X[idx], labels[idx], sample_size=None))
    return float(
        silhouette_score(X[mask], labels[mask], sample_size=sample_size, random_state=seed)
    )


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    _require_nontrivial_labels(labels)
    mask = labels != -1
    return float(davies_bouldin_score(X[mask], labels[mask]))


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    _require_nontrivial_labels(labels)
    mask = labels != -1
    return float(calinski_harabasz_score(X[mask], labels[mask]))


def dunn(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    diameter_cap: int = 2000,
    seed: int = 0,
) -> float:
    """Dunn index = min inter-cluster distance / max intra-cluster diameter.

    Higher is better. Robust to non-convex clusters.

    Implementation is O(c² · d + c · cap²) — feasible on DS-10 because c is
    small. Inter-cluster distance is the minimum *centroid* distance; diameter
    is the max pairwise distance within a cluster, computed on a per-cluster
    subsample of at most ``diameter_cap`` points so the dominant legitimate
    cluster (~283k) does not materialize an infeasible pairwise matrix. Noise
    points (label -1) are excluded.
    """
    _require_nontrivial_labels(labels)
    mask = labels != -1
    Xs, ls = X[mask], labels[mask]
    unique = np.unique(ls)
    if len(unique) < 2:
        raise ValueError("Dunn index requires >= 2 clusters")

    centroids = np.array([Xs[ls == c].mean(axis=0) for c in unique])
    inter_dists = pairwise_distances(centroids)
    np.fill_diagonal(inter_dists, np.inf)
    inter = float(inter_dists.min())

    rng = np.random.default_rng(seed)
    max_diam = 0.0
    for c in unique:
        members = Xs[ls == c]
        if len(members) < 2:
            continue
        if len(members) > diameter_cap:
            members = members[rng.choice(len(members), size=diameter_cap, replace=False)]
        diam = float(pairwise_distances(members).max())
        if diam > max_diam:
            max_diam = diam

    if max_diam == 0:
        return 0.0
    return float(inter / max_diam)


def ari(true: np.ndarray, pred: np.ndarray) -> float:
    return float(adjusted_rand_score(true, pred))


def nmi(true: np.ndarray, pred: np.ndarray) -> float:
    return float(normalized_mutual_info_score(true, pred))


def ami(true: np.ndarray, pred: np.ndarray) -> float:
    return float(adjusted_mutual_info_score(true, pred))


def v_measure(true: np.ndarray, pred: np.ndarray) -> float:
    return float(v_measure_score(true, pred))


def purity(true: np.ndarray, pred: np.ndarray) -> float:
    """Cluster purity = sum over clusters of (majority class count) / n.

    Labels of -1 (noise) are treated as their own cluster for this computation,
    matching the common convention in the clustering literature. ``true`` is
    expected to hold non-negative integer class labels (DS-10 ``Class`` ∈ {0,1}).
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    total = 0
    for c in np.unique(pred):
        members = true[pred == c]
        if len(members) == 0:
            continue
        total += int(np.max(np.bincount(members[members >= 0])))
    return float(total / len(true))


INTERNAL_METRICS: dict[str, Callable[..., float]] = {
    "silhouette": silhouette,
    "davies_bouldin": davies_bouldin,
    "calinski_harabasz": calinski_harabasz,
    "dunn": dunn,
}

EXTERNAL_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "ari": ari,
    "nmi": nmi,
    "ami": ami,
    "v_measure": v_measure,
    "purity": purity,
}

ALL_METRICS = {**INTERNAL_METRICS, **EXTERNAL_METRICS}


def score_internal(name: str, X: np.ndarray, labels: np.ndarray, **kw) -> float:
    if name not in INTERNAL_METRICS:
        raise KeyError(f"Unknown internal metric: {name!r}. Available: {list(INTERNAL_METRICS)}")
    return (
        INTERNAL_METRICS[name](X, labels, **kw)
        if name == "silhouette"
        else INTERNAL_METRICS[name](X, labels)
    )


def score_external(name: str, true: np.ndarray, pred: np.ndarray) -> float:
    if name not in EXTERNAL_METRICS:
        raise KeyError(f"Unknown external metric: {name!r}. Available: {list(EXTERNAL_METRICS)}")
    return EXTERNAL_METRICS[name](true, pred)


def higher_is_better(name: str) -> bool:
    """Direction convention for selecting the best k / model.

    Returns True when larger values are better. Davies-Bouldin is the only
    metric in the registry where lower is better.
    """
    return name != "davies_bouldin"
