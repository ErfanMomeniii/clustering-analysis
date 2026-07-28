import numpy as np
import pytest

from clustering_analysis.reduce import (
    fit_pca_for_variance,
    pca_explained_variance_curve,
    umap_embed,
)


@pytest.fixture
def rng_data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(500, 31))


def test_fit_pca_for_variance_returns_components_covering_target(rng_data):
    pca, n = fit_pca_for_variance(rng_data, target_variance=0.95)
    assert n <= 31
    cum = pca.explained_variance_ratio_.cumsum()[-1]
    assert cum >= 0.95


def test_fit_pca_with_too_high_target_returns_all_components(rng_data):
    _, n = fit_pca_for_variance(rng_data, target_variance=0.999999)
    assert n == 31


def test_pca_explained_variance_curve_length_matches_min(rng_data):
    curve = pca_explained_variance_curve(rng_data, max_components=20)
    assert len(curve) == 20
    assert curve[0] > 0
    assert curve[-1] <= 1.0
    assert (np.diff(curve) >= 0).all()


@pytest.mark.slow
def test_umap_embed_shape_matches_n_components(rng_data):
    emb = umap_embed(rng_data[:300], n_neighbors=15, min_dist=0.1, n_components=2, seed=0)
    assert emb.shape == (300, 2)
