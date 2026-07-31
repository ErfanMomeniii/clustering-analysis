import numpy as np
import pytest
from sklearn.datasets import make_blobs

from clustering_analysis.evaluation import (
    agreement_matrix,
    error_analysis,
    metric_ablation,
    performance_table,
    rank_runs,
    run_algorithm,
    score_run,
)

INTERNAL = ["silhouette", "davies_bouldin", "calinski_harabasz", "dunn"]
EXTERNAL = ["ari", "nmi", "ami", "v_measure", "purity"]


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0, n_features=5)
    return X, y


def _scored(X, y, name, k=3, **kw):
    run = run_algorithm(name, X, k, seed=0, **kw)
    return score_run(run, X, y, internal_metrics=INTERNAL, external_metrics=EXTERNAL, seed=0)


def test_run_algorithm_records_runtime_and_cluster_counts(blobs):
    X, _ = blobs
    run = run_algorithm("kmeans", X, 3, seed=0, n_init=5)
    assert run.runtime_s > 0
    assert run.n_clusters == 3
    assert run.n_noise == 0


def test_hdbscan_noise_is_counted_not_treated_as_a_cluster():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (100, 2)), rng.normal(8, 0.1, (100, 2)), rng.uniform(-20, 20, (40, 2))])
    run = run_algorithm("hdbscan", X, 2, seed=0, min_cluster_size=20, min_samples=5)
    assert run.n_noise > 0
    assert run.n_clusters == len(np.unique(run.labels[run.labels != -1]))


def test_performance_table_includes_runtime_and_every_metric(blobs):
    X, y = blobs
    row = performance_table([_scored(X, y, "kmeans", n_init=5)])[0]
    assert "runtime_s" in row
    for metric in INTERNAL + EXTERNAL:
        assert metric in row


def test_score_run_records_nan_for_degenerate_labelling(blobs):
    X, y = blobs
    run = run_algorithm("kmeans", X, 3, seed=0, n_init=5)
    run.labels = np.zeros(len(X), dtype=int)  # single cluster: internals undefined
    scored = score_run(run, X, y, internal_metrics=INTERNAL, external_metrics=EXTERNAL, seed=0)
    assert np.isnan(scored.internal["silhouette"])
    assert not np.isnan(scored.external["ari"])  # external metrics remain defined


def test_agreement_matrix_is_symmetric_with_unit_diagonal(blobs):
    X, y = blobs
    runs = [_scored(X, y, "kmeans", n_init=5), _scored(X, y, "ward"), _scored(X, y, "gmm")]
    names, M = agreement_matrix(runs)
    assert len(names) == 3
    np.testing.assert_allclose(M, M.T)
    np.testing.assert_allclose(np.diag(M), 1.0)


def test_agreement_is_one_for_identical_labellings(blobs):
    X, y = blobs
    run = _scored(X, y, "kmeans", n_init=5)
    _, M = agreement_matrix([run, run])
    assert M[0, 1] == pytest.approx(1.0)


def test_rank_runs_respects_metric_direction(blobs):
    X, y = blobs
    good = _scored(X, y, "kmeans", k=3, n_init=5)
    bad = _scored(X, y, "kmeans", k=15, n_init=5)
    assert rank_runs([bad, good], "silhouette")[0] is good  # higher is better
    assert rank_runs([bad, good], "davies_bouldin")[0] is good  # lower is better


def test_rank_runs_sorts_nan_scores_last(blobs):
    X, y = blobs
    ok = _scored(X, y, "kmeans", n_init=5)
    broken = _scored(X, y, "kmeans", n_init=5)
    broken.internal["silhouette"] = float("nan")
    assert rank_runs([broken, ok], "silhouette")[0] is ok


def test_error_analysis_flags_worst_points_and_compares_label_rate(blobs):
    X, y = blobs
    labels = run_algorithm("kmeans", X, 3, seed=0, n_init=5).labels
    rare = (np.arange(len(X)) % 50 == 0).astype(int)
    ea = error_analysis(X, labels, y_true=rare, n_worst=30, sample_size=None, seed=0)
    assert ea.n_inspected == 30
    assert 0.0 <= ea.frac_negative_silhouette <= 1.0
    assert len(ea.worst_indices) == 30
    assert ea.label_rate_overall == pytest.approx(rare.mean())
    assert sum(ea.per_cluster_counts.values()) == 30


def test_error_analysis_rejects_single_cluster(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match=">= 2 clusters"):
        error_analysis(X, np.zeros(len(X), dtype=int), sample_size=None)


def test_metric_ablation_covers_every_metric_with_reference_at_one(blobs):
    X, _ = blobs
    rows = metric_ablation(X[:150], 3, metrics=("euclidean", "manhattan", "cosine"))
    assert {r["metric"] for r in rows} == {"euclidean", "manhattan", "cosine"}
    reference = next(r for r in rows if r["metric"] == "euclidean")
    assert reference["ari_vs_reference"] == pytest.approx(1.0)


def test_metric_ablation_detects_a_metric_that_changes_the_clustering():
    """Cosine ignores magnitude, so radially-arranged blobs must cluster differently."""
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, 200)
    radii = rng.choice([1.0, 12.0], size=200)
    X = np.c_[radii * np.cos(angles), radii * np.sin(angles)]
    rows = metric_ablation(X, 2, metrics=("euclidean", "cosine"))
    cosine = next(r for r in rows if r["metric"] == "cosine")
    assert cosine["ari_vs_reference"] < 0.9
