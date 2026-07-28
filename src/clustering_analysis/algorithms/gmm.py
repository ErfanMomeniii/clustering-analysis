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
