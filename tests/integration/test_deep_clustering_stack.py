"""End-to-end tests for the deep clustering and interpretation stacks (slow).

Runs the three neural models (autoencoder + K-Means, DEC, DEC + InfoNCE) and the
interpretation chain (surrogate tree rules, then SHAP) on synthetic blobs, and
checks that the significance machinery wraps cleanly around the labels they
produce. Synthetic blobs are used deliberately: they have a known answer, so a
model that fails here is broken rather than merely facing hard data.
"""

import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

pytestmark = pytest.mark.slow


@pytest.fixture
def blobs_small():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.3, random_state=0, n_features=6)
    return X, y


def test_deep_clustering_pipeline_recovers_blobs(blobs_small):
    from clustering_analysis.deep import fit_ae_kmeans, fit_dec, fit_dec_infonce

    X, y = blobs_small
    lbl_ae, _, _, _ = fit_ae_kmeans(
        X, k=3, latent_dim=4, hidden_dims=[16], ae_epochs=20, ft_epochs=15, seed=0
    )
    lbl_dec, _, _, _ = fit_dec(
        X, k=3, latent_dim=4, hidden_dims=[16], init_epochs=20, dec_epochs=30, seed=0
    )
    lbl_ctn, _, _, _ = fit_dec_infonce(
        X,
        k=3,
        latent_dim=4,
        hidden_dims=[16],
        init_epochs=20,
        dec_epochs=30,
        contrastive_weight=0.1,
        seed=0,
    )
    assert adjusted_rand_score(y, lbl_ae) > 0.9
    assert adjusted_rand_score(y, lbl_dec) > 0.85
    assert adjusted_rand_score(y, lbl_ctn) > 0.85


def test_interpretation_pipeline_produces_rules_and_shap(blobs_small):
    from clustering_analysis.interpretation import extract_tree_rules, shap_attributions

    X, y = blobs_small
    names = [f"V{i}" for i in range(1, 7)]
    tree, rules = extract_tree_rules(X, y, feature_names=names, max_depth=3, min_samples_leaf=10)
    assert len(rules) > 0
    attr = shap_attributions(tree, X, feature_names=names, background=50)
    assert len(attr) >= 2


def test_significance_wraps_deep_clustering_labels(blobs_small):
    from clustering_analysis.deep import fit_dec
    from clustering_analysis.metrics import ari
    from clustering_analysis.significance import bootstrap_ci, permutation_test

    X, y = blobs_small
    labels, _, _, _ = fit_dec(
        X, k=3, latent_dim=4, hidden_dims=[16], init_epochs=15, dec_epochs=20, seed=0
    )
    ci = bootstrap_ci(ari, y, labels, n_bootstrap=100, seed=0)
    perm = permutation_test(ari, y, labels, n_permutations=100, seed=0)
    assert ci.lower <= ci.observed <= ci.upper
    assert perm.p_value < 0.1
