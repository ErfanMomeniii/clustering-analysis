import numpy as np
import pytest
from sklearn.datasets import make_blobs

from clustering_analysis.stability import (
    co_association_matrix,
    consensus_clustering,
    seed_stability,
    stability_summary,
)


@pytest.fixture
def blobs_k3():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.4, random_state=0, n_features=4)
    return X, y


def test_seed_stability_kmeans_high_on_separable_data(blobs_k3):
    X, _ = blobs_k3
    mean_ari, labels_list = seed_stability("kmeans", X, k=3, n_seeds=5, seed_offset=100)
    assert mean_ari > 0.8
    assert len(labels_list) == 5
    assert all(lbl.shape == (200,) for lbl in labels_list)


def test_seed_stability_deterministic_algorithm_is_one(blobs_k3):
    X, _ = blobs_k3
    mean_ari, _ = seed_stability("ward", X, k=3, n_seeds=3, seed_offset=0)
    assert mean_ari == pytest.approx(1.0)


def test_seed_stability_returns_one_for_single_seed(blobs_k3):
    X, _ = blobs_k3
    mean_ari, _ = seed_stability("kmeans", X, k=3, n_seeds=1, seed_offset=0)
    assert mean_ari == pytest.approx(1.0)  # no pairs to compare


def test_co_association_matrix_shape_and_diagonal(blobs_k3):
    X, _ = blobs_k3
    M = co_association_matrix("kmeans", X, k=3, n_resamples=5, sample_fraction=0.8, seed=0)
    assert M.shape == (200, 200)
    assert np.allclose(np.diag(M), 1.0)
    assert M.min() >= 0.0 and M.max() <= 1.0


def test_co_association_matrix_symmetric(blobs_k3):
    X, _ = blobs_k3
    M = co_association_matrix("kmeans", X, k=3, n_resamples=5, sample_fraction=0.8, seed=0)
    assert np.allclose(M, M.T, atol=1e-9)


def test_consensus_clustering_returns_k_labels(blobs_k3):
    X, _ = blobs_k3
    M = co_association_matrix("kmeans", X, k=3, n_resamples=10, sample_fraction=0.8, seed=0)
    labels = consensus_clustering(M, k=3, threshold=0.5)
    assert labels.shape == (200,)
    assert len(np.unique(labels)) == 3


def test_consensus_recovers_blob_structure(blobs_k3):
    X, y = blobs_k3
    M = co_association_matrix("kmeans", X, k=3, n_resamples=10, sample_fraction=0.8, seed=0)
    labels = consensus_clustering(M, k=3, threshold=0.5)
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(y, labels)
    assert ari > 0.8


def test_stability_summary_returns_expected_keys(blobs_k3):
    X, _ = blobs_k3
    s = stability_summary(
        "kmeans",
        X,
        k=3,
        n_seeds=3,
        seed_offset=0,
        n_resamples=5,
        sample_fraction=0.8,
        consensus_threshold=0.5,
        seed=0,
    )
    assert set(s) == {
        "algorithm",
        "k",
        "mean_seed_ari",
        "consensus_labels",
        "co_association",
        "n_consensus",
    }
    assert s["algorithm"] == "kmeans"
    assert s["k"] == 3
    assert 0.0 <= s["mean_seed_ari"] <= 1.0
    assert s["consensus_labels"].shape == (200,)
    assert s["n_consensus"] == 200


def test_stability_deterministic_for_fixed_seed(blobs_k3):
    X, _ = blobs_k3
    a = stability_summary("kmeans", X, k=3, n_seeds=3, n_resamples=5, seed=42)
    b = stability_summary("kmeans", X, k=3, n_seeds=3, n_resamples=5, seed=42)
    np.testing.assert_array_equal(a["consensus_labels"], b["consensus_labels"])


def test_co_association_matrix_subsamples_above_max_n():
    """Feasibility guard: large inputs are capped at max_n, not blown up to n×n."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 3))
    M = co_association_matrix(
        "kmeans", X, k=3, n_resamples=3, sample_fraction=0.8, seed=0, max_n=300
    )
    assert M.shape == (300, 300)


def test_co_association_matrix_preserves_small_input(blobs_k3):
    """Inputs under max_n are not subsampled — matrix is n×n."""
    X, _ = blobs_k3
    M = co_association_matrix(
        "kmeans", X, k=3, n_resamples=3, sample_fraction=0.8, seed=0, max_n=1000
    )
    assert M.shape == (200, 200)
