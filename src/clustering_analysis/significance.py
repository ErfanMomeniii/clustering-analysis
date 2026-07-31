"""Statistical significance: bootstrap CIs + permutation tests (Phase 3, bonus +3).

Two complementary procedures:

  - **bootstrap_ci**: resample the *cluster assignments* with replacement,
    recompute the metric each time, and report the percentile CI. This quantifies
    uncertainty in the metric value given the finite sample. We resample
    (true, pred) pairs jointly so the metric's sampling distribution is captured
    without re-fitting the clusterer (feasible on DS-10's 283k rows).

  - **permutation_test**: shuffle the predicted labels and recompute the
    external metric to build a null distribution. The p-value is the fraction
    of permutations whose metric meets or exceeds the observed value. This
    tests whether the cluster-structure ↔ label association is significantly
    stronger than chance.

Both are O(n) per resample and vectorised, so 1000 resamples on 283k rows is
feasible in well under a minute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIResult:
    metric: str
    observed: float
    mean: float
    std: float
    lower: float
    upper: float
    n_bootstrap: int
    level: float


@dataclass(frozen=True)
class PermutationResult:
    metric: str
    observed: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int


def bootstrap_ci(
    metric_fn,
    true: np.ndarray,
    pred: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> CIResult:
    """Bootstrap confidence interval for an external metric.

    ``metric_fn(true, pred) -> float``. Resamples (true, pred) pairs with
    replacement and reports the percentile CI at ``level``.
    """
    rng = np.random.default_rng(seed)
    n = len(true)
    true = np.asarray(true)
    pred = np.asarray(pred)
    observed = float(metric_fn(true, pred))
    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats[i] = metric_fn(true[idx], pred[idx])
    alpha = (1 - level) / 2
    lower, upper = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return CIResult(
        metric=metric_fn.__name__,
        observed=observed,
        mean=float(stats.mean()),
        std=float(stats.std()),
        lower=float(lower),
        upper=float(upper),
        n_bootstrap=n_bootstrap,
        level=level,
    )


def permutation_test(
    metric_fn,
    true: np.ndarray,
    pred: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> PermutationResult:
    """Permutation test for an external metric against a label-shuffle null.

    Shuffles ``pred`` and recomputes ``metric_fn(true, shuffled_pred)`` to build
    the null distribution. p-value = fraction of permutations >= observed
    (one-sided, since ARI/NMI/purity are higher-is-better).
    """
    rng = np.random.default_rng(seed)
    true = np.asarray(true)
    pred = np.asarray(pred)
    observed = float(metric_fn(true, pred))
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pred)
        null[i] = metric_fn(true, shuffled)
    p_value = float(np.mean(null >= observed))
    return PermutationResult(
        metric=metric_fn.__name__,
        observed=observed,
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        p_value=p_value,
        n_permutations=n_permutations,
    )


def compare_models(
    metric_fn,
    true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI on the *difference* of two models' metrics.

    Tests whether model A's metric is significantly different from model B's.
    Resamples (true, pred_a, pred_b) jointly and computes
    metric_a - metric_b each time; the CI on the difference tells us if one
    model is reliably better.
    """
    rng = np.random.default_rng(seed)
    true = np.asarray(true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(true)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric_fn(true[idx], preds_a[idx]) - metric_fn(true[idx], preds_b[idx])
    alpha = (1 - level) / 2
    lower, upper = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return {
        "metric": metric_fn.__name__,
        "observed_diff": float(metric_fn(true, preds_a) - metric_fn(true, preds_b)),
        "diff_mean": float(diffs.mean()),
        "diff_std": float(diffs.std()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "significant": bool(lower > 0 or upper < 0),
        "n_bootstrap": n_bootstrap,
        "level": level,
    }
