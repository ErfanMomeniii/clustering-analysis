import numpy as np
import pytest
from sklearn.datasets import make_blobs

from clustering_analysis.k_selection import (
    KSelectionResult,
    aggregate_k_selections,
    select_k_bic,
    select_k_bootstrap,
    select_k_elbow,
    select_k_gap,
    select_k_silhouette,
)


@pytest.fixture
def blobs_k3():
    X, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.5, random_state=0, n_features=4)
    return X


@pytest.fixture
def small_k_range():
    return [2, 3, 4, 5, 6]


def test_elbow_returns_result_with_scores(blobs_k3, small_k_range):
    r = select_k_elbow(blobs_k3, small_k_range, seed=0)
    assert isinstance(r, KSelectionResult)
    assert r.method == "elbow"
    assert r.recommended_k in small_k_range
    assert set(r.scores) == set(small_k_range)
    assert r.higher_is_better is False


def test_elbow_recommends_near_true_k(blobs_k3, small_k_range):
    r = select_k_elbow(blobs_k3, small_k_range, seed=0)
    assert r.recommended_k in (2, 3, 4)


def test_silhouette_picks_true_k(blobs_k3, small_k_range):
    r = select_k_silhouette(blobs_k3, small_k_range, seed=0, sample_size=200)
    assert r.recommended_k == 3
    assert r.higher_is_better is True


def test_silhouette_scores_decrease_then_increase(blobs_k3, small_k_range):
    r = select_k_silhouette(blobs_k3, small_k_range, seed=0, sample_size=200)
    # silhouette should be highest at k=3
    assert r.scores[3] >= r.scores[5]


def test_gap_picks_reasonable_k(blobs_k3, small_k_range):
    r = select_k_gap(blobs_k3, small_k_range, n_refs=5, seed=0)
    assert r.recommended_k in (2, 3, 4, 5, 6)
    assert r.higher_is_better is True


def test_bic_picks_true_k_or_lower(blobs_k3, small_k_range):
    r = select_k_bic(blobs_k3, small_k_range, seed=0)
    assert r.recommended_k in (2, 3, 4)
    assert r.higher_is_better is False


def test_bic_scores_decrease_with_overfit(blobs_k3, small_k_range):
    r = select_k_bic(blobs_k3, small_k_range, seed=0)
    # BIC should generally be lower at true k than heavily overfit
    assert r.scores[3] <= r.scores[6] + 2000


def test_bootstrap_picks_stable_k(blobs_k3, small_k_range):
    r = select_k_bootstrap(blobs_k3, small_k_range, n_resamples=5, sample_fraction=0.8, seed=0)
    assert r.recommended_k in (2, 3, 4)
    assert 0.0 <= r.scores[r.recommended_k] <= 1.0
    assert r.higher_is_better is True


def test_bootstrap_skips_k_below_2(blobs_k3):
    r = select_k_bootstrap(blobs_k3, [1, 2, 3], n_resamples=3, seed=0)
    assert 1 not in r.scores


def test_aggregate_votes_counts_each_method(blobs_k3, small_k_range):
    results = [
        select_k_silhouette(blobs_k3, small_k_range, seed=0, sample_size=200),
        select_k_bic(blobs_k3, small_k_range, seed=0),
    ]
    votes = aggregate_k_selections(results)
    assert sum(votes.values()) == 2
    assert all(k in small_k_range for k in votes)


def test_all_methods_deterministic_for_fixed_seed(blobs_k3, small_k_range):
    a = select_k_silhouette(blobs_k3, small_k_range, seed=11, sample_size=200)
    b = select_k_silhouette(blobs_k3, small_k_range, seed=11, sample_size=200)
    assert a.recommended_k == b.recommended_k


def test_gap_does_not_produce_inf_on_degenerate_data():
    """A degenerate reference (all points identical) gives inertia 0; the log
    floor must keep Gap finite rather than emitting -inf/+inf."""
    X = np.zeros((50, 2))  # all identical -> inertia 0 at any k>1
    r = select_k_gap(X, [2, 3], n_refs=3, seed=0)
    for v in r.scores.values():
        assert np.isfinite(v)
