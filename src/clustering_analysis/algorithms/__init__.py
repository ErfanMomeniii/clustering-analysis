"""Clustering algorithm registry (Phase 2).

Each algorithm exposes a uniform ``fit(X, k, *, seed) -> labels`` interface so
the evaluation loop can treat partitioning, hierarchical, density, and
model-based methods polymorphically. HDBSCAN ignores ``k`` (density-defined)
and returns -1 for noise.
"""

from clustering_analysis.algorithms.base import ALGORITHM_REGISTRY, fit_algorithm
from clustering_analysis.algorithms.gmm import fit_gmm
from clustering_analysis.algorithms.hdbscan import fit_hdbscan
from clustering_analysis.algorithms.kmeans import fit_kmeans
from clustering_analysis.algorithms.ward import fit_ward

__all__ = [
    "fit_algorithm",
    "ALGORITHM_REGISTRY",
    "fit_kmeans",
    "fit_ward",
    "fit_gmm",
    "fit_hdbscan",
]
