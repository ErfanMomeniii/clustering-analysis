"""End-to-end pipeline test on synthetic blobs.

Drives every Phase 1 module (features -> scaling -> reduce -> tendency)
against sklearn make_blobs, then runs K-Means as a sanity check and asserts
ARI > 0.95 between K-Means assignments and the true blob labels. This
proves the data layer + scaling + reduction stack is sound before we trust
it on DS-10.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering_analysis.reduce import fit_pca_for_variance
from clustering_analysis.scaling import build_scaler
from clustering_analysis.tendency import hopkins_statistic

pytestmark = pytest.mark.slow

V_COLS = [f"V{i}" for i in range(1, 29)]
ROBUST_COLS = ["log_amount", "time_sin", "time_cos"]


def _synthetic_processed_frame(n=600, seed=0):
    X, y = make_blobs(n_samples=n, centers=3, cluster_std=0.7, random_state=seed, n_features=28)
    df = pd.DataFrame(X, columns=V_COLS)
    rng = np.random.default_rng(seed)
    df["log_amount"] = rng.normal(loc=3, scale=1.5, size=n)
    df["time_sin"] = rng.uniform(-1, 1, size=n)
    df["time_cos"] = rng.uniform(-1, 1, size=n)
    return df, y


def test_pipeline_recovers_three_blobs():
    df, y = _synthetic_processed_frame()
    scaler = build_scaler("hybrid", v_features=V_COLS, robust_features=ROBUST_COLS)
    Xs = scaler.fit_transform(df)
    pca, n_pc = fit_pca_for_variance(Xs, target_variance=0.95)
    X_pca = pca.transform(Xs)
    H = hopkins_statistic(Xs, sample_size=60, seed=0)
    assert H > 0.7, f"Hopkins {H:.3f} too low on synthetic blobs"
    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(X_pca)
    ari = adjusted_rand_score(y, km.labels_)
    assert ari > 0.95, f"ARI {ari:.3f} below 0.95 — pipeline regression"
