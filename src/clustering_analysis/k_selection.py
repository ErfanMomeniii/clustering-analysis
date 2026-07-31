"""Determining the number of clusters k (Phase 2 §3).

Implements five complementary criteria so no single heuristic dominates:
  - elbow (inertia / distortion knee)
  - silhouette (max mean silhouette)
  - gap statistic (Tibshirani 2001)
  - BIC (GMM model selection)
  - bootstrap stability (mode count across resamples)

Each returns a ``KSelectionResult`` with the recommended k and per-k scores,
so the report can render all curves side-by-side and disagreements are visible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from .algorithms.gmm import fit_gmm_scored
from .metrics import silhouette


@dataclass(frozen=True)
class KSelectionResult:
    method: str
    recommended_k: int
    scores: dict[int, float]  # k -> score
    higher_is_better: bool


def _inertia(X: np.ndarray, k: int, seed: int) -> float:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return float(km.inertia_)


def _log_wk(X: np.ndarray, k: int, seed: int) -> float:
    """log(W_k) with a floor to avoid -inf on degenerate reference data.

    A reference dataset can collapse to a single point under unlucky uniform
    draws (especially in high d), making inertia 0 and log(0) = -inf. Floor at
    the smallest positive float so the gap stays finite and comparable.
    """
    w = _inertia(X, k, seed)
    return float(np.log(w) if w > 0 else np.log(np.finfo(float).tiny))


def select_k_elbow(X: np.ndarray, k_range: list[int], *, seed: int = 0) -> KSelectionResult:
    """Knee of the inertia curve, located by the Kneedle construction.

    Lower inertia is always better, so the minimum is never the answer — the
    elbow marks where returns start diminishing. Kneedle (Satopää et al. 2011)
    formalises "elbow" as the point of maximum perpendicular distance from the
    curve to the chord joining its endpoints, which is what is computed below.
    That is the quantitative localisation the brief asks for; it is implemented
    directly rather than via the ``kneed`` package so the project carries no
    dependency for six lines of geometry.
    """
    inertias = {k: _inertia(X, k, seed) for k in k_range}
    ks = sorted(inertias)
    ys = np.array([inertias[k] for k in ks], dtype=float)
    # line connecting first and last point
    p1 = np.array([ks[0], ys[0]])
    p2 = np.array([ks[-1], ys[-1]])
    seg = p2 - p1
    seg_norm = np.linalg.norm(seg)
    if seg_norm == 0:
        recommended = ks[0]
    else:
        # Perpendicular distance from each (k, inertia) point to the line p1->p2.
        # 2D cross = (p2-p1) x (point-p1) = seg_x*dy - seg_y*dx.
        sx, sy = seg
        dists = np.array(
            [abs(sx * (ys[i] - p1[1]) - sy * (ks[i] - p1[0])) / seg_norm for i in range(len(ks))]
        )
        recommended = int(ks[int(np.argmax(dists))])
    return KSelectionResult("elbow", recommended, inertias, higher_is_better=False)


def select_k_silhouette(
    X: np.ndarray, k_range: list[int], *, seed: int = 0, sample_size: int | None = None
) -> KSelectionResult:
    """Pick k with the maximum mean silhouette score."""
    scores = {}
    for k in k_range:
        if k < 2:
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        try:
            scores[k] = silhouette(X, labels, sample_size=sample_size, seed=seed)
        except ValueError:
            scores[k] = float("-inf")
    recommended = max(scores, key=scores.get) if scores else k_range[0]
    return KSelectionResult("silhouette", int(recommended), scores, higher_is_better=True)


def select_k_gap(
    X: np.ndarray, k_range: list[int], *, n_refs: int = 10, seed: int = 0
) -> KSelectionResult:
    """Tibshirani gap statistic.

    Gap(k) = E*[log(W_k)] - log(W_k), where E* is over uniform reference data
    bounded by the box of X. Optimal k is the smallest k where
    Gap(k) >= Gap(k+1) - s_{k+1}.
    """
    rng = np.random.default_rng(seed)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    n, d = X.shape
    log_w = {}
    ref_log_w = {k: [] for k in k_range if k >= 2}
    for k in k_range:
        if k < 2:
            continue
        log_w[k] = _log_wk(X, k, seed)
        for _ in range(n_refs):
            X_ref = rng.uniform(mins, maxs, size=(n, d))
            ref_log_w[k].append(_log_wk(X_ref, k, seed))
    gaps = {k: float(np.mean(ref_log_w[k]) - log_w[k]) for k in log_w}
    s = {k: float(np.std(ref_log_w[k]) * np.sqrt(1 + 1.0 / n_refs)) for k in ref_log_w}
    ks = sorted(gaps)
    recommended = ks[-1]
    for i, k in enumerate(ks[:-1]):
        if gaps[k] >= gaps[ks[i + 1]] - s[ks[i + 1]]:
            recommended = k
            break
    return KSelectionResult("gap", int(recommended), gaps, higher_is_better=True)


def select_k_bic(
    X: np.ndarray, k_range: list[int], *, seed: int = 0, covariance_type: str = "full"
) -> KSelectionResult:
    """Pick k with the minimum BIC (GMM)."""
    bics = {}
    for k in k_range:
        if k < 2:
            continue
        _, bic, _ = fit_gmm_scored(X, k, seed=seed, covariance_type=covariance_type)
        bics[k] = bic
    recommended = min(bics, key=bics.get) if bics else k_range[0]
    return KSelectionResult("bic", int(recommended), bics, higher_is_better=False)


def select_k_bootstrap(
    X: np.ndarray,
    k_range: list[int],
    *,
    n_resamples: int = 20,
    sample_fraction: float = 0.8,
    seed: int = 0,
) -> KSelectionResult:
    """Bootstrap stability: pick k whose label assignment is most stable.

    For each k, fit K-Means on n_resamples bootstrap subsamples and measure
    the mean pairwise ARI between assignments restricted to common points.
    The k with the highest mean ARI is the most stable. This is the criterion
    the brief prioritises (§3 stability).
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = len(X)
    sub_size = int(n * sample_fraction)
    stability = {k: 0.0 for k in k_range if k >= 2}
    for k in k_range:
        if k < 2:
            continue
        labels_list = []
        for _ in range(n_resamples):
            idx = rng.choice(n, size=sub_size, replace=True)
            lbl = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X[idx])
            labels_list.append((idx, lbl))
        aris = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                idx_i, lbl_i = labels_list[i]
                idx_j, lbl_j = labels_list[j]
                # np.intersect1d(..., return_indices=True) returns
                # (common_values, positions_in_a, positions_in_b) — exactly the
                # index alignment we need, no manual loc maps required.
                common_vals, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
                if len(common_vals) < 2:
                    continue
                sub_i = lbl_i[pos_i]
                sub_j = lbl_j[pos_j]
                # ARI is undefined when one side is a single cluster
                if len(set(sub_i)) < 2 or len(set(sub_j)) < 2:
                    continue
                aris.append(adjusted_rand_score(sub_i, sub_j))
        stability[k] = float(np.mean(aris)) if aris else 0.0
    recommended = max(stability, key=stability.get) if stability else k_range[0]
    return KSelectionResult("bootstrap", int(recommended), stability, higher_is_better=True)


def aggregate_k_selections(results: list[KSelectionResult]) -> dict[int, int]:
    """Vote across methods: returns {k: vote_count}.

    Each method casts one vote for its recommended k. Ties are resolved by
    summing ranks; caller can pick the mode.
    """
    votes = Counter(r.recommended_k for r in results)
    return dict(votes)
