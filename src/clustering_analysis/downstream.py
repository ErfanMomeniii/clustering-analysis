"""Cluster-based downstream analysis (Phase 3 §4.3).

Two of the three analyses the brief offers, both chosen because DS-10 supports
them cleanly:

  - **cluster-conditional modelling**: fit one logistic model per cluster and
    compare against a single global model on the same holdout. If the clusters
    carry structure, the per-cluster models should beat the global one; if they
    do not, the clustering is decorative. Average precision is the headline
    score because at a 0.17 % positive rate ROC-AUC flatters everything.

  - **cluster-based anomaly detection**: rank points by distance to their own
    cluster centroid and check whether the top of that ranking is enriched for
    fraud. This is the unsupervised fraud detector the research question asks
    for — it consumes the clustering, never the label.

Temporal cluster evolution is not implemented: DS-10 spans two days, which is
too short a window for a merge/split analysis to mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ConditionalModellingResult:
    n_train: int
    n_test: int
    global_average_precision: float
    clustered_average_precision: float
    global_roc_auc: float
    clustered_roc_auc: float
    per_cluster: list[dict]
    lift: float


def _fit_scorer(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression | None:
    """Fit a logistic model, or return None when a fold has one class only."""
    if len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(X, y)
    return model


def cluster_conditional_modelling(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> ConditionalModellingResult:
    """Compare per-cluster models against one global model on a shared holdout.

    The split is stratified on ``y`` so both sides carry positives. A cluster
    whose training fold has a single class falls back to the global model's
    prediction for its test rows, which is the honest comparison: the clustered
    approach gets no credit for cases it cannot model.
    """
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_fraction, random_state=seed, stratify=y
    )

    global_model = _fit_scorer(X[train_idx], y[train_idx], seed)
    if global_model is None:
        raise ValueError("Global model needs both classes in the training split")
    global_scores = global_model.predict_proba(X[test_idx])[:, 1]

    clustered_scores = np.array(global_scores, dtype=float, copy=True)
    per_cluster = []
    for c in np.unique(clusters):
        tr = train_idx[clusters[train_idx] == c]
        te_mask = clusters[test_idx] == c
        if te_mask.sum() == 0:
            continue
        local = _fit_scorer(X[tr], y[tr], seed) if len(tr) > 10 else None
        if local is not None:
            clustered_scores[te_mask] = local.predict_proba(X[test_idx][te_mask])[:, 1]
        per_cluster.append(
            {
                "cluster": int(c),
                "n_train": int(len(tr)),
                "n_test": int(te_mask.sum()),
                "positives_train": int(y[tr].sum()),
                "positives_test": int(y[test_idx][te_mask].sum()),
                "modelled": local is not None,
            }
        )

    y_test = y[test_idx]
    g_ap = float(average_precision_score(y_test, global_scores))
    c_ap = float(average_precision_score(y_test, clustered_scores))
    return ConditionalModellingResult(
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        global_average_precision=g_ap,
        clustered_average_precision=c_ap,
        global_roc_auc=float(roc_auc_score(y_test, global_scores)),
        clustered_roc_auc=float(roc_auc_score(y_test, clustered_scores)),
        per_cluster=per_cluster,
        lift=float(c_ap / g_ap) if g_ap > 0 else float("nan"),
    )


@dataclass(frozen=True)
class AnomalyRanking:
    n_points: int
    n_flagged: int
    top_fraction: float
    precision_at_k: float
    recall_at_k: float
    base_rate: float
    enrichment: float
    roc_auc: float
    average_precision: float
    flagged_indices: np.ndarray


def centroid_distances(X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Distance from each point to its own cluster's centroid.

    Noise points (label -1) get the distance to the nearest cluster centroid:
    a density algorithm already judged them anomalous, so they must not be
    silently dropped from an anomaly ranking.
    """
    labels = np.unique(clusters)
    real = labels[labels != -1]
    centroids = {int(c): X[clusters == c].mean(axis=0) for c in real}
    dist = np.empty(len(X), dtype=float)
    for c, mu in centroids.items():
        mask = clusters == c
        dist[mask] = np.linalg.norm(X[mask] - mu, axis=1)
    noise = clusters == -1
    if noise.any():
        C = np.stack([centroids[int(c)] for c in real])
        dist[noise] = np.linalg.norm(X[noise][:, None, :] - C[None, :, :], axis=2).min(axis=1)
    return dist


def cluster_anomaly_detection(
    X: np.ndarray,
    clusters: np.ndarray,
    y: np.ndarray | None = None,
    *,
    top_fraction: float = 0.01,
) -> AnomalyRanking:
    """Rank points by distance to their cluster centroid and score the ranking.

    ``enrichment`` is precision@k divided by the base rate — the factor by which
    reviewing the flagged tail beats reviewing transactions at random, which is
    the number an actual fraud team would care about.
    """
    dist = centroid_distances(X, clusters)
    n_flagged = max(1, int(round(top_fraction * len(X))))
    order = np.argsort(-dist)
    flagged = order[:n_flagged]

    if y is None:
        return AnomalyRanking(
            n_points=len(X),
            n_flagged=n_flagged,
            top_fraction=top_fraction,
            precision_at_k=float("nan"),
            recall_at_k=float("nan"),
            base_rate=float("nan"),
            enrichment=float("nan"),
            roc_auc=float("nan"),
            average_precision=float("nan"),
            flagged_indices=flagged,
        )

    y = np.asarray(y)
    base = float(y.mean())
    prec = float(y[flagged].mean())
    return AnomalyRanking(
        n_points=len(X),
        n_flagged=n_flagged,
        top_fraction=top_fraction,
        precision_at_k=prec,
        recall_at_k=float(y[flagged].sum() / y.sum()) if y.sum() else float("nan"),
        base_rate=base,
        enrichment=float(prec / base) if base > 0 else float("nan"),
        roc_auc=float(roc_auc_score(y, dist)),
        average_precision=float(average_precision_score(y, dist)),
        flagged_indices=flagged,
    )
