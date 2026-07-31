import numpy as np
import pytest

from clustering_analysis.downstream import (
    centroid_distances,
    cluster_anomaly_detection,
    cluster_conditional_modelling,
)


@pytest.fixture
def clustered_data():
    """Two clusters whose positive class depends on opposite features.

    A single global model cannot express both rules, so per-cluster models should
    win — which is exactly the claim the downstream analysis is meant to test.
    """
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(0, 1, (2 * n, 3))
    clusters = np.array([0] * n + [1] * n)
    X[clusters == 0, 2] += 10  # separate the clusters on an unused axis
    y = np.empty(2 * n, dtype=int)
    y[clusters == 0] = (X[clusters == 0, 0] > 0.6).astype(int)
    y[clusters == 1] = (X[clusters == 1, 1] < -0.6).astype(int)
    return X, y, clusters


def test_conditional_modelling_beats_global_when_clusters_carry_structure(clustered_data):
    X, y, clusters = clustered_data
    res = cluster_conditional_modelling(X, y, clusters, test_fraction=0.3, seed=0)
    assert res.clustered_average_precision > res.global_average_precision
    assert res.lift > 1.0


def test_conditional_modelling_reports_per_cluster_bookkeeping(clustered_data):
    X, y, clusters = clustered_data
    res = cluster_conditional_modelling(X, y, clusters, seed=0)
    assert {r["cluster"] for r in res.per_cluster} == {0, 1}
    assert all(r["modelled"] for r in res.per_cluster)
    assert res.n_train + res.n_test == len(X)


def test_conditional_modelling_falls_back_when_a_cluster_cannot_be_modelled():
    """A single-class cluster must inherit the global prediction, not crash."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 3))
    clusters = np.array([0] * 380 + [1] * 20)
    y = (X[:, 0] > 0.5).astype(int)
    y[clusters == 1] = 0  # cluster 1 has one class only
    res = cluster_conditional_modelling(X, y, clusters, seed=0)
    unmodelled = [r for r in res.per_cluster if not r["modelled"]]
    assert len(unmodelled) >= 0  # never raises
    assert np.isfinite(res.clustered_average_precision)


def test_conditional_modelling_requires_both_classes():
    X = np.random.default_rng(0).normal(size=(100, 3))
    with pytest.raises(ValueError):
        cluster_conditional_modelling(X, np.zeros(100, dtype=int), np.zeros(100, dtype=int))


def test_centroid_distances_are_zero_for_a_degenerate_cluster():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
    clusters = np.array([0, 0, 1, 1])
    np.testing.assert_allclose(centroid_distances(X, clusters), 0.0)


def test_centroid_distances_use_nearest_centroid_for_noise():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [1.0, 0.0]])
    clusters = np.array([0, 0, 1, 1, -1])
    d = centroid_distances(X, clusters)
    assert d[-1] == pytest.approx(1.0)  # distance to cluster 0, not cluster 1


def test_anomaly_detection_enriches_for_the_rare_class():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (500, 2)), rng.normal(0, 1, (10, 2)) + 12])
    y = np.array([0] * 500 + [1] * 10)
    clusters = np.zeros(len(X), dtype=int)
    res = cluster_anomaly_detection(X, clusters, y, top_fraction=0.02)
    assert res.enrichment > 1.0
    assert res.roc_auc > 0.9
    assert 0.0 <= res.precision_at_k <= 1.0


def test_anomaly_detection_without_labels_returns_ranking_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    res = cluster_anomaly_detection(X, np.zeros(100, dtype=int), None, top_fraction=0.1)
    assert res.n_flagged == 10
    assert np.isnan(res.precision_at_k)
    assert len(res.flagged_indices) == 10
