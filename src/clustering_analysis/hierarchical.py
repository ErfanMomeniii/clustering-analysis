"""Hierarchical clustering diagnostics (Phase 2, Group B).

The Group-B requirement is broader than "run Ward": the brief asks for all four
standard linkage criteria compared on the same distance matrix, a cophenetic
correlation per linkage, and at least two strategies for cutting the
dendrogram into a flat clustering.

Feasibility: every routine here consumes a condensed pairwise distance vector,
which is O(n²). On DS-10 (283,726 rows) that is ~322 GB, so the caller must
pass a subsample — ``dataset.stratified_subsample`` provides one that retains
every fraud row. This is a deliberate protocol choice, not an oversight: the
dendrogram is a diagnostic over the geometry, and a stratified 20k-row sample
preserves that geometry while fitting in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist

from .metrics import silhouette

LINKAGES = ("single", "complete", "average", "ward")

# scipy's pdist and sklearn use different names for the same metrics. The rest of
# the package speaks sklearn (``metric_ablation``, ``AgglomerativeClustering``),
# so translate at this boundary instead of forcing callers to remember which
# vocabulary each module wants.
_SCIPY_METRIC_ALIASES = {"manhattan": "cityblock", "l1": "cityblock", "l2": "euclidean"}


def _scipy_metric(metric: str) -> str:
    return _SCIPY_METRIC_ALIASES.get(metric, metric)


@dataclass(frozen=True)
class LinkageDiagnostics:
    """Per-linkage dendrogram diagnostics at a fixed k."""

    linkage: str
    cophenetic_corr: float
    labels: np.ndarray
    n_clusters: int
    max_merge_height: float
    cluster_sizes: list[int]


def cophenetic_correlation(Z: np.ndarray, distances: np.ndarray) -> float:
    """Correlation between input distances and dendrogram merge heights.

    Near 1.0 means the tree faithfully represents the pairwise distances; a low
    value means the dendrogram distorts the geometry it claims to summarise, so
    any flat clustering cut from it is suspect.
    """
    corr, _ = cophenet(Z, distances)
    return float(corr)


def build_linkages(
    X: np.ndarray,
    *,
    metric: str = "euclidean",
    linkages: tuple[str, ...] = LINKAGES,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute one linkage matrix per criterion from a single distance vector.

    Returns ``({name: Z}, condensed_distances)``. Ward is only defined for
    Euclidean geometry, so it is skipped for any other metric rather than
    silently producing a meaningless tree.
    """
    distances = pdist(X, metric=_scipy_metric(metric))
    out: dict[str, np.ndarray] = {}
    for name in linkages:
        if name == "ward" and _scipy_metric(metric) != "euclidean":
            continue
        out[name] = linkage(distances, method=name)
    return out, distances


def linkage_comparison(
    X: np.ndarray,
    k: int,
    *,
    metric: str = "euclidean",
    linkages: tuple[str, ...] = LINKAGES,
    trees: dict[str, np.ndarray] | None = None,
    distances: np.ndarray | None = None,
) -> list[LinkageDiagnostics]:
    """Compare linkage criteria at a fixed k on one shared distance matrix.

    ``trees``/``distances`` may be passed in when the caller has already run
    ``build_linkages``; the condensed distance vector is O(n²), so recomputing
    it is the single most expensive avoidable step in the phase.
    """
    if trees is None or distances is None:
        trees, distances = build_linkages(X, metric=metric, linkages=linkages)
    results = []
    for name, Z in trees.items():
        labels = fcluster(Z, t=k, criterion="maxclust").astype(int)
        sizes = sorted((int((labels == c).sum()) for c in np.unique(labels)), reverse=True)
        results.append(
            LinkageDiagnostics(
                linkage=name,
                cophenetic_corr=cophenetic_correlation(Z, distances),
                labels=labels,
                n_clusters=int(len(np.unique(labels))),
                max_merge_height=float(Z[:, 2].max()),
                cluster_sizes=sizes,
            )
        )
    return results


def cut_at_height(Z: np.ndarray, *, height: float) -> np.ndarray:
    """Cutting strategy 1: a fixed-height horizontal cut of the dendrogram."""
    return fcluster(Z, t=height, criterion="distance").astype(int)


def cut_at_height_fraction(Z: np.ndarray, *, fraction: float = 0.7) -> tuple[np.ndarray, float]:
    """Fixed-height cut expressed as a fraction of the tallest merge.

    Returns ``(labels, height)``. Anchoring the height to the tree's own scale
    keeps the strategy comparable across linkages, whose merge heights are not
    on a common scale.
    """
    height = float(fraction * Z[:, 2].max())
    return cut_at_height(Z, height=height), height


def cut_by_max_silhouette(
    Z: np.ndarray,
    X: np.ndarray,
    k_range: list[int],
    *,
    sample_size: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, int, dict[int, float]]:
    """Cutting strategy 2: choose the k whose flat clustering maximises silhouette.

    Returns ``(labels, best_k, {k: silhouette})``. Cuts that collapse to a
    single cluster score -inf so they can never win.
    """
    scores: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_range:
        if k < 2:
            continue
        labels = fcluster(Z, t=k, criterion="maxclust").astype(int)
        labels_by_k[k] = labels
        try:
            scores[k] = silhouette(X, labels, sample_size=sample_size, seed=seed)
        except ValueError:
            scores[k] = float("-inf")
    best_k = max(scores, key=scores.get)
    return labels_by_k[best_k], int(best_k), scores
