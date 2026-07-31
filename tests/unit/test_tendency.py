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


def test_hopkins_null_control_returns_half_on_structureless_data():
    """The control is what makes a saturated H=1.0 headline defensible."""
    from clustering_analysis.tendency import hopkins_null_control

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (300, 6)), rng.normal(20, 0.1, (300, 6))])
    control = hopkins_null_control(X, sample_size=60, seed=0, n_repeats=3)
    assert 0.4 <= control["null_h_mean"] <= 0.6
    assert control["n_repeats"] == 3
    assert control["shape"] == (600, 6)


def test_hopkins_control_is_far_below_the_clustered_value():
    from clustering_analysis.tendency import hopkins_null_control, hopkins_statistic

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.05, (300, 6)), rng.normal(20, 0.05, (300, 6))])
    clustered = hopkins_statistic(X, sample_size=60, seed=0)
    control = hopkins_null_control(X, sample_size=60, seed=0, n_repeats=2)
    assert clustered > control["null_h_mean"] + 0.3


def test_hopkins_control_is_deterministic_for_a_fixed_seed():
    from clustering_analysis.tendency import hopkins_null_control

    X = np.random.default_rng(1).normal(size=(200, 4))
    a = hopkins_null_control(X, sample_size=40, seed=7, n_repeats=2)
    b = hopkins_null_control(X, sample_size=40, seed=7, n_repeats=2)
    assert a["null_h_values"] == b["null_h_values"]
