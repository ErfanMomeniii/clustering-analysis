"""Drift monitoring for the production clustering pipeline (Phase 3 §4.5).

Two standard statistics per feature, computed between the data a model was fit
on and a later period:

  - **PSI** (Population Stability Index): sum over bins of
    ``(late - early) * ln(late / early)``. The industry convention is that PSI
    below 0.1 is stable, 0.1-0.25 warrants attention, and above 0.25 means the
    population has moved. The re-fit trigger lives in
    ``params.yaml::phase3.drift.psi_threshold`` so the threshold is documented
    rather than folklore.

  - **Kolmogorov-Smirnov**: the maximum gap between the two empirical CDFs, with
    a p-value. KS catches shape changes that a binned statistic can miss.

Bin edges come from the *early* (reference) period only — deriving them from the
combined data would leak the later distribution into the reference and shrink
the drift being measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

EPSILON = 1e-6  # keeps empty bins from producing an infinite log ratio


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """PSI between two samples of one feature, binned on reference quantiles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # a near-constant feature cannot drift in distribution
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_frac = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_frac = np.histogram(current, bins=edges)[0] / len(current)
    ref_frac = np.clip(ref_frac, EPSILON, None)
    cur_frac = np.clip(cur_frac, EPSILON, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str],
    *,
    n_bins: int = 10,
    psi_threshold: float = 0.2,
) -> list[FeatureDrift]:
    """Per-feature PSI + KS between a reference and a current matrix."""
    if reference.shape[1] != current.shape[1]:
        raise ValueError(
            f"Feature count mismatch: reference has {reference.shape[1]}, "
            f"current has {current.shape[1]}"
        )
    out = []
    for i, name in enumerate(feature_names):
        psi = population_stability_index(reference[:, i], current[:, i], n_bins=n_bins)
        ks = ks_2samp(reference[:, i], current[:, i])
        out.append(
            FeatureDrift(
                feature=name,
                psi=psi,
                ks_statistic=float(ks.statistic),
                ks_pvalue=float(ks.pvalue),
                drifted=bool(psi >= psi_threshold),
            )
        )
    return sorted(out, key=lambda d: -d.psi)


def temporal_split(
    X: np.ndarray,
    time_values: np.ndarray,
    *,
    split_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into early / late halves along a time axis.

    Ordering by time (rather than sampling at random) is the point: drift is a
    question about *later* data, so a random split would answer nothing.
    """
    order = np.argsort(time_values, kind="stable")
    cut = int(len(order) * split_fraction)
    return X[order[:cut]], X[order[cut:]]


def drift_report(drifts: list[FeatureDrift], *, psi_threshold: float) -> dict:
    """Summarise per-feature drift into a re-fit decision."""
    psis = [d.psi for d in drifts]
    flagged = [d.feature for d in drifts if d.drifted]
    return {
        "n_features": len(drifts),
        "psi_threshold": psi_threshold,
        "max_psi": float(max(psis)) if psis else 0.0,
        "mean_psi": float(np.mean(psis)) if psis else 0.0,
        "n_drifted": len(flagged),
        "drifted_features": flagged,
        "refit_required": bool(flagged),
    }
