"""Gaussian Mixture Model (model-based family).

Returns hard assignments via ``fit_predict``; BIC/AIC are exposed through
``fit_gmm_scored`` so k-selection can use them directly without re-fitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from .base import register


@register("gmm")
def fit_gmm(X: np.ndarray, k: int, *, seed: int, covariance_type: str = "full", **_) -> np.ndarray:
    if k < 2:
        raise ValueError("GMM requires k >= 2")
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=seed,
        reg_covar=1e-6,
    )
    return gmm.fit_predict(X).astype(int)


def fit_gmm_scored(X: np.ndarray, k: int, *, seed: int, covariance_type: str = "full"):
    """Fit GMM and return (labels, bic, aic) for k-selection."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=seed,
        reg_covar=1e-6,
    )
    labels = gmm.fit_predict(X).astype(int)
    return labels, float(gmm.bic(X)), float(gmm.aic(X))


COVARIANCE_TYPES = ("spherical", "diag", "tied", "full")


def compare_covariance_types(
    X: np.ndarray,
    k: int,
    *,
    seed: int,
    covariance_types: tuple[str, ...] = COVARIANCE_TYPES,
) -> list[dict]:
    """Fit one GMM per covariance structure and report model-selection scores.

    The brief requires at least three covariance structures compared with their
    log-likelihood trajectories. ``n_parameters`` is included because it is what
    makes BIC/AIC differ from raw log-likelihood: a ``full`` covariance buys
    likelihood with O(k·d²) parameters, and BIC is the check on whether that
    trade is worth it.

    Returns one dict per structure, sorted by BIC (lower is better).
    """
    rows = []
    for cov in covariance_types:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov,
            random_state=seed,
            reg_covar=1e-6,
        )
        labels = gmm.fit_predict(X).astype(int)
        rows.append(
            {
                "covariance_type": cov,
                "log_likelihood": float(gmm.score(X) * len(X)),
                "mean_log_likelihood": float(gmm.score(X)),
                "bic": float(gmm.bic(X)),
                "aic": float(gmm.aic(X)),
                "n_parameters": int(gmm._n_parameters()),
                "n_iter": int(gmm.n_iter_),
                "converged": bool(gmm.converged_),
                "n_clusters_used": int(len(np.unique(labels))),
            }
        )
    return sorted(rows, key=lambda r: r["bic"])
