import numpy as np

from clustering_analysis.tendency import hopkins_statistic, vat_ordering


def test_hopkins_on_uniform_random_near_half():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(2000, 5))
    H = hopkins_statistic(X, sample_size=200, seed=0)
    assert 0.35 < H < 0.65


def test_hopkins_on_well_separated_blobs_above_threshold():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=2000, centers=4, cluster_std=0.5, random_state=0)
    H = hopkins_statistic(X, sample_size=200, seed=0)
    assert H > 0.75


def test_hopkins_deterministic_for_fixed_seed():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(2000, 5))
    a = hopkins_statistic(X, sample_size=200, seed=42)
    b = hopkins_statistic(X, sample_size=200, seed=42)
    assert a == b


def test_vat_ordering_returns_permutation_of_indices():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    order = vat_ordering(X)
    assert sorted(order) == list(range(100))


def test_vat_ordering_produces_diagonal_block_on_blobs():
    from scipy.spatial.distance import pdist, squareform
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.4, random_state=0)
    order = vat_ordering(X)
    D = squareform(pdist(X[order]))
    block_dist = float(D[:40, :40].mean())
    off_block_dist = float(D[:40, 80:].mean())
    assert block_dist < off_block_dist
