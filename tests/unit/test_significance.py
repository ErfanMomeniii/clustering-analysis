import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from clustering_analysis.metrics import ari, nmi
from clustering_analysis.significance import (
    CIResult,
    PermutationResult,
    bootstrap_ci,
    compare_models,
    permutation_test,
)


@pytest.fixture
def data():
    X, y = make_blobs(n_samples=400, centers=3, cluster_std=0.4, random_state=0, n_features=5)
    pred_good = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    rng = np.random.default_rng(1)
    pred_random = rng.integers(0, 3, size=len(y))
    return y, pred_good, pred_random


def test_bootstrap_ci_returns_correct_fields(data):
    y, pred, _ = data
    r = bootstrap_ci(ari, y, pred, n_bootstrap=200, level=0.95, seed=0)
    assert isinstance(r, CIResult)
    assert r.metric == "ari"
    assert 0.0 <= r.observed <= 1.0
    assert r.lower <= r.observed <= r.upper
    assert r.n_bootstrap == 200
    assert r.level == 0.95


def test_bootstrap_ci_higher_level_shrinks_interval(data):
    y, pred, _ = data
    r95 = bootstrap_ci(ari, y, pred, n_bootstrap=500, level=0.95, seed=0)
    r99 = bootstrap_ci(ari, y, pred, n_bootstrap=500, level=0.99, seed=0)
    assert (r99.upper - r99.lower) >= (r95.upper - r95.lower)


def test_bootstrap_ci_deterministic_for_fixed_seed(data):
    y, pred, _ = data
    a = bootstrap_ci(ari, y, pred, n_bootstrap=100, seed=42)
    b = bootstrap_ci(ari, y, pred, n_bootstrap=100, seed=42)
    assert a.lower == b.lower
    assert a.upper == b.upper


def test_bootstrap_ci_on_perfect_assignment_has_tight_interval():
    y = np.array([0, 0, 1, 1, 2, 2] * 50)
    r = bootstrap_ci(ari, y, y, n_bootstrap=200, seed=0)
    assert r.observed == pytest.approx(1.0)
    assert r.lower > 0.95


def test_permutation_test_returns_correct_fields(data):
    y, pred, _ = data
    r = permutation_test(ari, y, pred, n_permutations=200, seed=0)
    assert isinstance(r, PermutationResult)
    assert r.metric == "ari"
    assert r.observed > 0.5
    assert 0.0 <= r.p_value <= 0.1  # strong association => low p


def test_permutation_test_on_random_assignment_high_p_value(data):
    y, _, pred_random = data
    r = permutation_test(ari, y, pred_random, n_permutations=200, seed=0)
    assert r.p_value > 0.2  # no association => high p


def test_permutation_test_on_perfect_assignment_p_value_zero():
    y = np.array([0, 0, 1, 1, 2, 2] * 50)
    r = permutation_test(ari, y, y, n_permutations=200, seed=0)
    assert r.p_value < 0.01


def test_permutation_test_deterministic_for_fixed_seed(data):
    y, pred, _ = data
    a = permutation_test(ari, y, pred, n_permutations=100, seed=7)
    b = permutation_test(ari, y, pred, n_permutations=100, seed=7)
    assert a.p_value == b.p_value


def test_compare_models_detects_significant_difference(data):
    y, pred_good, pred_random = data
    r = compare_models(ari, y, pred_good, pred_random, n_bootstrap=500, seed=0)
    assert r["metric"] == "ari"
    assert r["observed_diff"] > 0.5
    assert r["significant"] is True
    assert r["ci_lower"] > 0


def test_compare_models_on_identical_predictions_not_significant(data):
    y, pred_good, _ = data
    r = compare_models(ari, y, pred_good, pred_good, n_bootstrap=200, seed=0)
    assert r["observed_diff"] == pytest.approx(0.0)
    assert r["significant"] is False
    assert r["ci_lower"] <= 0 <= r["ci_upper"]


def test_significance_works_with_multiple_metrics(data):
    y, pred, _ = data
    for fn in (ari, nmi):
        r = bootstrap_ci(fn, y, pred, n_bootstrap=100, seed=0)
        assert r.lower <= r.observed <= r.upper
        p = permutation_test(fn, y, pred, n_permutations=100, seed=0)
        assert p.p_value < 0.1
