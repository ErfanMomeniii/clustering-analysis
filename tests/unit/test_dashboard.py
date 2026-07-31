import json

import numpy as np
import pytest

from clustering_analysis.dashboard import (
    assign_to_nearest_cluster,
    cluster_centroids,
    cluster_profile_df,
    load_summary,
)


def test_cluster_profile_df_has_one_row_per_cluster():
    X = np.random.RandomState(0).randn(90, 3)
    clusters = np.array([0] * 30 + [1] * 30 + [2] * 30)
    labels = np.zeros(90, dtype=int)
    df = cluster_profile_df(X, clusters, labels, ["a", "b", "c"])
    assert len(df) == 3
    assert set(df["cluster"]) == {0, 1, 2}
    assert (df["size"] == 30).all()


def test_cluster_profile_df_fraud_rate_in_unit_interval():
    rng = np.random.RandomState(0)
    X = rng.randn(100, 2)
    clusters = rng.randint(0, 3, 100)
    labels = rng.randint(0, 2, 100)
    df = cluster_profile_df(X, clusters, labels, ["a", "b"])
    assert (df["fraud_rate"] >= 0).all()
    assert (df["fraud_rate"] <= 1).all()


def test_cluster_profile_df_skips_noise_label():
    X = np.random.RandomState(0).randn(50, 2)
    clusters = np.array([0] * 20 + [1] * 20 + [-1] * 10)
    labels = np.zeros(50, dtype=int)
    df = cluster_profile_df(X, clusters, labels, ["a", "b"])
    assert -1 not in df["cluster"].values
    assert len(df) == 2


def test_cluster_profile_df_feature_means_match():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    clusters = np.array([0, 0, 1, 1])
    labels = np.array([0, 1, 0, 1])
    df = cluster_profile_df(X, clusters, labels, ["a", "b"])
    row0 = df[df["cluster"] == 0].iloc[0]
    assert row0["a"] == 2.0 and row0["b"] == 3.0
    assert row0["fraud_rate"] == 0.5


# --- live-assignment page (brief §4.6 requirement 4) ----------------------- #
def test_cluster_centroids_excludes_noise_and_matches_means():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [99.0, 99.0]])
    clusters = np.array([0, 0, 1, -1])
    ids, centroids = cluster_centroids(X, clusters)
    assert ids.tolist() == [0, 1]
    np.testing.assert_allclose(centroids[0], [1.0, 0.0])
    np.testing.assert_allclose(centroids[1], [10.0, 10.0])


def test_cluster_centroids_rejects_an_all_noise_labelling():
    with pytest.raises(ValueError, match="No non-noise clusters"):
        cluster_centroids(np.zeros((5, 2)), np.full(5, -1))


def test_assign_to_nearest_cluster_picks_the_closest_centroid():
    ids = np.array([0, 1, 2])
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    assigned, table = assign_to_nearest_cluster(np.array([9.0, 0.5]), ids, centroids)
    assert assigned == 1
    assert table["cluster"].iloc[0] == 1
    assert list(table["rank"]) == [1, 2, 3]


def test_assign_to_nearest_cluster_reports_distance_to_every_centroid():
    ids = np.array([0, 1])
    centroids = np.array([[0.0, 0.0], [3.0, 4.0]])
    _, table = assign_to_nearest_cluster(np.zeros(2), ids, centroids)
    assert len(table) == 2
    assert table.set_index("cluster").loc[1, "distance"] == pytest.approx(5.0)


def test_assign_to_nearest_cluster_rejects_a_wrong_width_record():
    with pytest.raises(ValueError, match="Record has 2 features"):
        assign_to_nearest_cluster(np.zeros(2), np.array([0]), np.zeros((1, 5)))


def test_load_summary_returns_none_when_a_phase_has_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr("clustering_analysis.dashboard.REPO_ROOT", tmp_path)
    assert load_summary("phase2") is None


def test_load_summary_reads_a_persisted_phase_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("clustering_analysis.dashboard.REPO_ROOT", tmp_path)
    target = tmp_path / "results" / "phase2"
    target.mkdir(parents=True)
    (target / "summary.json").write_text(json.dumps({"selected_k": 4}))
    assert load_summary("phase2") == {"selected_k": 4}
