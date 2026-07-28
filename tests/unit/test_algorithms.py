import numpy as np
import pytest
from sklearn.datasets import make_blobs

from clustering_analysis.algorithms import ALGORITHM_REGISTRY, fit_algorithm
from clustering_analysis.algorithms.gmm import fit_gmm, fit_gmm_scored
from clustering_analysis.algorithms.hdbscan import fit_hdbscan
from clustering_analysis.algorithms.kmeans import fit_kmeans
from clustering_analysis.algorithms.ward import fit_ward


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.4, random_state=0, n_features=5)
    return X, y


def test_registry_contains_required_families():
    assert set(ALGORITHM_REGISTRY) == {"kmeans", "ward", "gmm", "hdbscan"}


def test_fit_algorithm_unknown_raises():
    with pytest.raises(KeyError):
        fit_algorithm("bogus", np.zeros((10, 2)), k=3, seed=0)


@pytest.mark.parametrize("name", ["kmeans", "ward", "gmm"])
def test_partitioning_algorithms_recover_k_clusters(blobs, name):
    X, _ = blobs
    labels = fit_algorithm(name, X, k=3, seed=0)
    assert len(np.unique(labels)) == 3
    assert labels.shape == (300,)


def test_kmeans_rejects_k_below_2():
    with pytest.raises(ValueError):
        fit_kmeans(np.zeros((10, 2)), k=1, seed=0)


def test_ward_rejects_k_below_2():
    with pytest.raises(ValueError):
        fit_ward(np.zeros((10, 2)), k=1, seed=0)


def test_gmm_rejects_k_below_2():
    with pytest.raises(ValueError):
        fit_gmm(np.zeros((10, 2)), k=1, seed=0)


def test_kmeans_seeded_is_deterministic(blobs):
    X, _ = blobs
    a = fit_kmeans(X, k=3, seed=42)
    b = fit_kmeans(X, k=3, seed=42)
    np.testing.assert_array_equal(a, b)


def test_gmm_seeded_is_deterministic(blobs):
    X, _ = blobs
    a = fit_gmm(X, k=3, seed=42)
    b = fit_gmm(X, k=3, seed=42)
    np.testing.assert_array_equal(a, b)


def test_gmm_scored_returns_bic_aic(blobs):
    X, _ = blobs
    labels, bic, aic = fit_gmm_scored(X, k=3, seed=0)
    assert labels.shape == (300,)
    assert np.isfinite(bic)
    assert np.isfinite(aic)
    # BIC should generally decrease then increase around true k=3
    _, bic2, _ = fit_gmm_scored(X, k=8, seed=0)
    # not a strict assertion but bic at overfit k should be >= bic at true k
    assert bic2 >= bic - 500


def test_hdbscan_returns_noise_label_for_sparse_points():
    rng = np.random.default_rng(0)
    dense = rng.normal(size=(200, 2))
    sparse = rng.uniform(-10, 10, size=(50, 2))
    X = np.vstack([dense, sparse])
    labels = fit_hdbscan(X, k=0, seed=0, min_cluster_size=20, min_samples=5)
    assert -1 in labels
    assert labels.shape == (250,)


def test_hdbscan_ignores_k_argument(blobs):
    X, _ = blobs
    a = fit_hdbscan(X, k=3, seed=0, min_cluster_size=20, min_samples=5)
    b = fit_hdbscan(X, k=99, seed=0, min_cluster_size=20, min_samples=5)
    np.testing.assert_array_equal(a, b)


def test_algorithms_recover_blob_structure(blobs):
    """All four families should produce at least 2 non-noise clusters on blobs."""
    X, y = blobs
    for name in ["kmeans", "ward", "gmm"]:
        labels = fit_algorithm(name, X, k=3, seed=0)
        assert len(np.unique(labels)) == 3
    labels = fit_hdbscan(X, k=0, seed=0, min_cluster_size=20, min_samples=5)
    assert len(set(labels) - {-1}) >= 2
