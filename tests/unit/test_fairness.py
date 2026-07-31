import numpy as np
import pytest

from clustering_analysis.fairness import (
    audit_cluster_composition,
    preprocessing_sensitivity,
    quantile_strata,
)


def test_quantile_strata_splits_into_requested_bins():
    values = np.arange(400, dtype=float)
    strata = quantile_strata(values, n_strata=4)
    assert len(np.unique(strata)) == 4
    counts = np.bincount(strata)
    assert counts.min() >= 95 and counts.max() <= 105


def test_quantile_strata_collapses_on_tied_values():
    """A constant feature has no quantile edges; the audit must degrade, not crash."""
    strata = quantile_strata(np.ones(100), n_strata=4)
    assert len(np.unique(strata)) == 1


def test_audit_reports_ratio_one_when_cluster_mirrors_population():
    strata = np.tile([0, 1, 2, 3], 100)
    clusters = np.repeat([0, 1], 200)
    audit = audit_cluster_composition(clusters, strata, attribute="test")
    assert audit.max_representation_ratio == pytest.approx(1.0, abs=0.05)
    assert audit.concentrated_clusters == []


def test_audit_flags_a_cluster_that_captures_one_stratum():
    strata = np.repeat([0, 1, 2, 3], 100)
    clusters = np.where(strata == 0, 1, 0)  # cluster 1 is entirely stratum 0
    audit = audit_cluster_composition(clusters, strata, attribute="test")
    assert 1 in audit.concentrated_clusters
    assert audit.max_representation_ratio == pytest.approx(4.0, rel=0.01)


def test_audit_records_one_row_per_cluster_with_shares():
    strata = np.tile([0, 1], 200)
    clusters = np.repeat([0, 1, 2, 3], 100)
    audit = audit_cluster_composition(clusters, strata, attribute="test")
    assert len(audit.per_cluster) == 4
    for row in audit.per_cluster:
        shares = [v for k, v in row.items() if k.startswith("share_")]
        assert sum(shares) == pytest.approx(1.0)


def test_sensitivity_reports_ari_one_for_an_unchanged_representation():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.4, random_state=0, n_features=4)
    reference = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    results = preprocessing_sensitivity({"identity": (X, "same matrix")}, reference, 3, seed=0)
    assert results[0].ari_vs_reference == pytest.approx(1.0)


def test_sensitivity_detects_a_destructive_preprocessing_variant():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.4, random_state=0, n_features=4)
    reference = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    noise = np.random.default_rng(0).normal(size=X.shape) * 50
    results = preprocessing_sensitivity(
        {"noise": (noise, "structure destroyed")}, reference, 3, seed=0
    )
    assert results[0].ari_vs_reference < 0.2
