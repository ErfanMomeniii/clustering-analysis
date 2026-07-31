import numpy as np
import pytest

from clustering_analysis.drift import (
    drift_report,
    feature_drift,
    population_stability_index,
    temporal_split,
)


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert population_stability_index(a, b) < 0.05


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    small = population_stability_index(ref, rng.normal(0.2, 1, 5000))
    large = population_stability_index(ref, rng.normal(2.0, 1, 5000))
    assert small < large
    assert large > 0.25  # "population has moved" by the usual convention


def test_psi_is_zero_for_a_constant_feature():
    """A near-constant feature yields fewer than two bin edges; PSI must be 0, not NaN."""
    assert population_stability_index(np.ones(100), np.ones(100)) == 0.0


def test_psi_is_finite_when_a_bin_empties_completely():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    shifted = rng.normal(50, 1, 1000)  # no overlap at all
    psi = population_stability_index(ref, shifted)
    assert np.isfinite(psi) and psi > 1.0


def test_feature_drift_ranks_the_drifted_feature_first():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(2000, 3))
    cur = ref.copy()
    cur[:, 1] += 3.0
    drifts = feature_drift(ref, cur, ["a", "b", "c"], psi_threshold=0.2)
    assert drifts[0].feature == "b"
    assert drifts[0].drifted
    assert not drifts[-1].drifted


def test_feature_drift_rejects_mismatched_feature_counts():
    with pytest.raises(ValueError, match="Feature count mismatch"):
        feature_drift(np.zeros((10, 3)), np.zeros((10, 4)), ["a", "b", "c"])


def test_ks_pvalue_is_small_for_a_real_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(1000, 1))
    cur = rng.normal(1.5, 1, (1000, 1))
    d = feature_drift(ref, cur, ["a"])[0]
    assert d.ks_pvalue < 0.01
    assert d.ks_statistic > 0.3


def test_temporal_split_orders_by_time_not_by_row():
    X = np.arange(10).reshape(-1, 1).astype(float)
    times = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    early, late = temporal_split(X, times, split_fraction=0.5)
    assert early.ravel().tolist() == [9, 8, 7, 6, 5]
    assert late.ravel().tolist() == [4, 3, 2, 1, 0]


def test_drift_report_triggers_a_refit_only_when_a_feature_breaches_threshold():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(2000, 2))
    stable = drift_report(feature_drift(ref, rng.normal(size=(2000, 2)), ["a", "b"]), psi_threshold=0.2)
    assert not stable["refit_required"]

    cur = ref.copy()
    cur[:, 0] += 4.0
    moved = drift_report(feature_drift(ref, cur, ["a", "b"], psi_threshold=0.2), psi_threshold=0.2)
    assert moved["refit_required"]
    assert moved["drifted_features"] == ["a"]
