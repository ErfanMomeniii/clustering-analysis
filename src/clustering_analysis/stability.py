"""Stability analysis + consensus clustering (Phase 2 §3).

Two complementary notions of stability:
  - **seed stability**: re-fit an algorithm under N seeds and measure pairwise
    ARI on the full dataset. High mean ARI => the solution is not a fluke of
    initialisation (matters for K-Means and GMM; Ward and HDBSCAN are
    deterministic).
  - **consensus via co-association**: across bootstrap resamples, build the
    n×n co-association matrix (fraction of times two points cluster together),
    then re-cluster that affinity matrix to produce a single stable labelling.

Feasibility: the co-association matrix is O(n²) in memory and is therefore
built on a configurable subsample (Monti et al. 2003 routinely cap at a few
thousand points). On DS-10 (283,726 rows) the default ``max_n`` of 2000 keeps
the matrix at ~32 MB.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from .algorithms import fit_algorithm


def seed_stability(
    name: str,
    X: np.ndarray,
    k: int,
    *,
    n_seeds: int,
    seed_offset: int = 0,
    **fit_kwargs,
) -> tuple[float, list[np.ndarray]]:
    """Mean pairwise ARI across n_seeds refits with different seeds.

    Returns (mean_ari, list_of_label_vectors). Deterministic algorithms
    (ward, hdbscan) will trivially return mean_ari == 1.0 because the seed is
    ignored — that is correct and signals deterministic stability.
    """
    labels_list = [
        fit_algorithm(name, X, k, seed=seed_offset + i, **fit_kwargs) for i in range(n_seeds)
    ]
    aris = [
        adjusted_rand_score(labels_list[i], labels_list[j])
        for i in range(len(labels_list))
        for j in range(i + 1, len(labels_list))
    ]
    return float(np.mean(aris)) if aris else 1.0, labels_list


def co_association_matrix(
    name: str,
    X: np.ndarray,
    k: int,
    *,
    n_resamples: int,
    sample_fraction: float,
    seed: int = 0,
    max_n: int = 2000,
    **fit_kwargs,
) -> np.ndarray:
    """Build the co-association matrix from bootstrap resamples.

    Entry (i, j) = fraction of resamples where i and j were both drawn and
    assigned to the same cluster. Pairs never co-drawn default to 0.0.

    To keep memory O(n²) feasible on DS-10, the matrix is built on a fixed
    random subsample of at most ``max_n`` points (drawn once, before
    bootstrapping). All subsequent bootstrap resamples are drawn from this
    submatrix, matching the Monti et al. (2003) consensus-clustering protocol.

    A **fixed** seed is used for every fit so that the only source of
    variation across resamples is the bootstrap sampling itself — not
    initialisation noise. This isolates the stability signal we want.
    """
    n = len(X)
    if n > max_n:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max_n, replace=False)
        X = X[keep]
        n = max_n
    sub_size = max(2, int(n * sample_fraction))
    co = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=int)
    rng = np.random.default_rng(seed)
    # Fixed fit seed: isolate bootstrap-induced variation, not init noise.
    for _ in range(n_resamples):
        idx = rng.choice(n, size=sub_size, replace=True)
        labels = fit_algorithm(name, X[idx], k, seed=seed, **fit_kwargs)
        same = labels[:, None] == labels[None, :]
        co[np.ix_(idx, idx)] += same
        counts[np.ix_(idx, idx)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(counts > 0, co / counts, 0.0)
    np.fill_diagonal(M, 1.0)
    return M


def consensus_clustering(
    co_association: np.ndarray,
    k: int,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Re-cluster the co-association matrix to produce stable consensus labels.

    Binarise the affinity at ``threshold`` then run average-link agglomerative
    clustering on the 1 - affinity precomputed distance. Returns a single label
    vector of length n.

    Average link is used (not Ward) because Ward cannot consume a precomputed
    distance matrix; average link is the standard choice for consensus
    clustering (Monti et al. 2003) and is robust to chain effects.
    """
    affinity = np.clip(co_association, 0.0, 1.0)
    affinity_bin = (affinity >= threshold).astype(float)
    # symmetrise to guard against bootstrap sampling asymmetry
    affinity_bin = np.maximum(affinity_bin, affinity_bin.T)
    np.fill_diagonal(affinity_bin, 1.0)
    distance = 1.0 - affinity_bin
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    model = AgglomerativeClustering(n_clusters=k, linkage="average", metric="precomputed")
    return model.fit_predict(distance).astype(int)


def stability_summary(
    name: str,
    X: np.ndarray,
    k: int,
    *,
    n_seeds: int,
    seed_offset: int = 0,
    n_resamples: int = 20,
    sample_fraction: float = 0.8,
    consensus_threshold: float = 0.5,
    max_n: int = 2000,
    seed: int = 0,
    **fit_kwargs,
) -> dict:
    """Bundle seed stability + consensus into a single report dict.

    Keys: ``algorithm``, ``k``, ``mean_seed_ari``, ``consensus_labels``,
    ``co_association``, ``n_consensus``. Used by the Phase 2 pipeline stage.
    """
    mean_ari, _ = seed_stability(name, X, k, n_seeds=n_seeds, seed_offset=seed_offset, **fit_kwargs)
    M = co_association_matrix(
        name,
        X,
        k,
        n_resamples=n_resamples,
        sample_fraction=sample_fraction,
        seed=seed,
        max_n=max_n,
        **fit_kwargs,
    )
    consensus = consensus_clustering(M, k, threshold=consensus_threshold)
    return {
        "algorithm": name,
        "k": k,
        "mean_seed_ari": mean_ari,
        "consensus_labels": consensus,
        "co_association": M,
        "n_consensus": len(consensus),
    }
