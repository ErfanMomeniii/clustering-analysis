import numpy as np
import pytest
from sklearn.datasets import make_blobs

from clustering_analysis.hierarchical import (
    LINKAGES,
    build_linkages,
    cophenetic_correlation,
    cut_at_height,
    cut_at_height_fraction,
    cut_by_max_silhouette,
    linkage_comparison,
)


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=0, n_features=4)
    return X, y


def test_all_four_linkages_are_compared(blobs):
    X, _ = blobs
    diags = linkage_comparison(X, 3)
    assert {d.linkage for d in diags} == set(LINKAGES)


def test_cophenetic_correlation_in_unit_interval(blobs):
    X, _ = blobs
    for d in linkage_comparison(X, 3):
        assert -1.0 <= d.cophenetic_corr <= 1.0


def test_average_linkage_has_the_highest_cophenetic_correlation(blobs):
    """UPGMA maximises cophenetic correlation by construction — a useful sanity check."""
    X, _ = blobs
    trees, distances = build_linkages(X)
    corrs = {name: cophenetic_correlation(Z, distances) for name, Z in trees.items()}
    assert corrs["average"] == max(corrs.values())
    assert corrs["average"] > 0.7


@pytest.mark.parametrize("metric", ["manhattan", "cityblock", "cosine"])
def test_ward_skipped_for_non_euclidean_metric(blobs, metric):
    """sklearn spells it "manhattan", scipy "cityblock"; both must work."""
    X, _ = blobs
    trees, _ = build_linkages(X, metric=metric)
    assert "ward" not in trees
    assert {"single", "complete", "average"} <= set(trees)


def test_ward_kept_for_euclidean_aliases(blobs):
    X, _ = blobs
    trees, _ = build_linkages(X, metric="l2")
    assert "ward" in trees


def test_linkage_comparison_reuses_precomputed_trees(blobs):
    X, _ = blobs
    trees, distances = build_linkages(X)
    a = linkage_comparison(X, 3, trees=trees, distances=distances)
    b = linkage_comparison(X, 3)
    assert [d.linkage for d in a] == [d.linkage for d in b]
    for x, y in zip(a, b, strict=True):
        assert x.cophenetic_corr == pytest.approx(y.cophenetic_corr)


def test_fixed_height_cut_produces_fewer_clusters_at_greater_height(blobs):
    X, _ = blobs
    trees, _ = build_linkages(X, linkages=("ward",))
    Z = trees["ward"]
    low = cut_at_height(Z, height=0.2 * Z[:, 2].max())
    high = cut_at_height(Z, height=0.9 * Z[:, 2].max())
    assert len(np.unique(high)) < len(np.unique(low))


def test_cut_at_height_fraction_reports_the_height_used(blobs):
    X, _ = blobs
    trees, _ = build_linkages(X, linkages=("ward",))
    labels, height = cut_at_height_fraction(trees["ward"], fraction=0.5)
    assert height == pytest.approx(0.5 * trees["ward"][:, 2].max())
    assert len(labels) == len(X)


def test_max_silhouette_cut_recovers_true_cluster_count(blobs):
    X, _ = blobs
    trees, _ = build_linkages(X, linkages=("ward",))
    labels, best_k, scores = cut_by_max_silhouette(trees["ward"], X, [2, 3, 4, 5, 6])
    assert best_k == 3
    assert len(np.unique(labels)) == 3
    assert scores[best_k] == max(scores.values())


def test_two_cutting_strategies_are_independent(blobs):
    """The two strategies must be able to disagree, else only one was implemented."""
    X, _ = blobs
    trees, _ = build_linkages(X, linkages=("ward",))
    Z = trees["ward"]
    height_labels, _ = cut_at_height_fraction(Z, fraction=0.95)
    sil_labels, _, _ = cut_by_max_silhouette(Z, X, [2, 3, 4, 5])
    assert len(np.unique(height_labels)) != len(np.unique(sil_labels))
