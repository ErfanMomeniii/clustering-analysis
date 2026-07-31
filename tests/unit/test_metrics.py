import numpy as np
import pytest
from sklearn.datasets import make_blobs

from clustering_analysis.metrics import (
    EXTERNAL_METRICS,
    INTERNAL_METRICS,
    dunn,
    higher_is_better,
    purity,
    score_external,
    score_internal,
)


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0, n_features=5)
    return X, y


def test_internal_metric_names_match_brief():
    assert set(INTERNAL_METRICS) == {"silhouette", "davies_bouldin", "calinski_harabasz", "dunn"}


def test_external_metric_names_match_brief():
    assert set(EXTERNAL_METRICS) == {"ari", "nmi", "ami", "v_measure", "purity"}


def test_silhouette_on_blobs_is_high(blobs):
    X, y = blobs
    assert score_internal("silhouette", X, y) > 0.4


def test_davies_bouldin_on_blobs_is_low(blobs):
    X, y = blobs
    assert score_internal("davies_bouldin", X, y) < 1.0


def test_calinski_harabasz_positive(blobs):
    X, y = blobs
    assert score_internal("calinski_harabasz", X, y) > 0


def test_dunn_on_well_separated_blobs_positive(blobs):
    X, y = blobs
    assert dunn(X, y) > 0


def test_dunn_handles_large_cluster_without_materializing_nx2():
    """Feasibility: the dominant cluster is subsampled for diameter, not exploded."""
    rng = np.random.default_rng(0)
    big = rng.normal(loc=0, scale=0.1, size=(5000, 3))  # one huge tight cluster
    c2 = rng.normal(loc=10, scale=0.1, size=(50, 3))
    c3 = rng.normal(loc=20, scale=0.1, size=(50, 3))
    X = np.vstack([big, c2, c3])
    labels = np.concatenate([np.zeros(5000), np.ones(50), np.full(50, 2)]).astype(int)
    val = dunn(X, labels, diameter_cap=200, seed=0)
    assert np.isfinite(val) and val > 0


def test_dunn_diameter_cap_seed_is_deterministic():
    rng = np.random.default_rng(0)
    big = rng.normal(loc=0, scale=0.1, size=(2000, 3))
    c2 = rng.normal(loc=10, scale=0.1, size=(50, 3))
    X = np.vstack([big, c2])
    labels = np.concatenate([np.zeros(2000), np.ones(50)]).astype(int)
    a = dunn(X, labels, diameter_cap=100, seed=7)
    b = dunn(X, labels, diameter_cap=100, seed=7)
    assert a == b


def test_dunn_lower_is_better_direction():
    assert higher_is_better("dunn") is True
    assert higher_is_better("davies_bouldin") is False
    assert higher_is_better("silhouette") is True


def test_internal_metrics_reject_single_cluster(blobs):
    X, _ = blobs
    labels = np.zeros(len(X), dtype=int)
    with pytest.raises(ValueError):
        score_internal("silhouette", X, labels)


def test_internal_metrics_ignore_noise_points(blobs):
    X, y = blobs
    labels = y.copy()
    labels[:10] = -1
    s = score_internal("silhouette", X, labels, sample_size=200, seed=0)
    assert -1.0 <= s <= 1.0


def test_silhouette_subsample_is_deterministic(blobs):
    X, y = blobs
    a = score_internal("silhouette", X, y, sample_size=100, seed=7)
    b = score_internal("silhouette", X, y, sample_size=100, seed=7)
    assert a == b


def test_external_metrics_perfect_assignment(blobs):
    _, y = blobs
    assert score_external("ari", y, y) == pytest.approx(1.0)
    assert score_external("nmi", y, y) == pytest.approx(1.0)
    assert score_external("ami", y, y) == pytest.approx(1.0)
    assert score_external("v_measure", y, y) == pytest.approx(1.0)
    assert score_external("purity", y, y) == pytest.approx(1.0)


def test_external_metrics_random_assignment_near_zero(blobs):
    _, y = blobs
    rng = np.random.default_rng(0)
    perm = rng.permutation(y)
    assert score_external("ari", y, perm) <= 0.05


def test_purity_on_mixed_cluster():
    true = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 0, 0, 1, 0, 1])
    # cluster 0: {0,0,1,0} -> majority 0 count 3
    # cluster 1: {1,1} -> majority 1 count 2
    # purity = (3 + 2) / 6
    assert purity(true, pred) == pytest.approx(5 / 6)


def test_score_internal_forwards_dunn_kwargs():
    """A shared kwargs bag must reach dunn, not be silently dropped."""
    X, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.4, random_state=0, n_features=4)
    labels = np.repeat([0, 1, 2, 0], 100)
    # a cap of 3 truncates every cluster, so it must not equal the uncapped value
    capped = dunn(X, labels, diameter_cap=3, seed=3)
    uncapped = dunn(X, labels, diameter_cap=400, seed=3)
    assert capped != pytest.approx(uncapped), "fixture too weak to detect a dropped kwarg"

    routed = score_internal("dunn", X, labels, diameter_cap=3, seed=3)
    assert routed == pytest.approx(capped)  # kwargs reached dunn
    assert routed != pytest.approx(uncapped)  # and were not silently dropped


def test_score_internal_ignores_kwargs_a_metric_cannot_accept():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.4, random_state=0, n_features=4)
    # davies_bouldin takes neither sample_size nor diameter_cap
    assert score_internal("davies_bouldin", X, y, sample_size=50, diameter_cap=10) > 0


def test_score_internal_unknown_raises():
    with pytest.raises(KeyError):
        score_internal("bogus", np.zeros((5, 2)), np.zeros(5, dtype=int))


def test_score_external_unknown_raises():
    with pytest.raises(KeyError):
        score_external("bogus", np.zeros(5), np.zeros(5))


def test_higher_is_better_unknown_defaults_true():
    assert higher_is_better("unknown_future_metric") is True
