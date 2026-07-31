import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier

from clustering_analysis.interpretation import (
    TreeRule,
    extract_tree_rules,
    shap_attributions,
)


@pytest.fixture
def blobs_with_names():
    X, y = make_blobs(n_samples=400, centers=3, cluster_std=0.5, random_state=0, n_features=6)
    feature_names = [f"V{i}" for i in range(1, 7)]
    return X, y, feature_names


def test_extract_tree_rules_reports_real_cluster_labels_not_class_indices():
    """Cluster ids need not be 0..k-1 — e.g. after dropping HDBSCAN noise."""
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.3, random_state=0, n_features=4)
    remap = {0: 0, 1: 3, 2: 7}
    labels = np.array([remap[v] for v in y])
    _, rules = extract_tree_rules(X, labels, max_depth=3, min_samples_leaf=10)
    assert {r.cluster for r in rules}.issubset({0, 3, 7})
    assert {r.cluster for r in rules} == {0, 3, 7}


def test_extract_tree_rules_support_counts_samples_not_proportions():
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.3, random_state=0, n_features=4)
    _, rules = extract_tree_rules(X, y, max_depth=3, min_samples_leaf=10)
    # sklearn >= 1.4 normalises tree_.value; support must still be a sample count
    assert all(r.support >= 10 for r in rules)
    assert sum(r.support for r in rules) == 600


def test_extract_tree_rules_returns_tree_and_rules(blobs_with_names):
    X, y, names = blobs_with_names
    tree, rules = extract_tree_rules(
        X, y, feature_names=names, max_depth=3, min_samples_leaf=10, seed=0
    )
    assert isinstance(tree, DecisionTreeClassifier)
    assert len(rules) > 0
    assert all(isinstance(r, TreeRule) for r in rules)
    assert all(r.support > 0 for r in rules)


def test_extract_tree_rules_conditions_reference_feature_names(blobs_with_names):
    X, y, names = blobs_with_names
    _, rules = extract_tree_rules(X, y, feature_names=names, max_depth=2, min_samples_leaf=10)
    for rule in rules:
        for cond in rule.conditions:
            assert any(fn in cond for fn in names)


def test_extract_tree_rules_precision_in_unit_interval(blobs_with_names):
    X, y, names = blobs_with_names
    _, rules = extract_tree_rules(X, y, feature_names=names, max_depth=3, min_samples_leaf=10)
    for r in rules:
        assert 0.0 <= r.precision <= 1.0


def test_extract_tree_rules_ignores_noise_points():
    X, y = make_blobs(n_samples=300, centers=2, cluster_std=0.3, random_state=0, n_features=4)
    y_with_noise = y.copy()
    y_with_noise[:20] = -1
    _, rules = extract_tree_rules(X, y_with_noise, max_depth=2, min_samples_leaf=10)
    assert all(r.cluster >= 0 for r in rules)


def test_extract_tree_rules_deterministic(blobs_with_names):
    X, y, names = blobs_with_names
    _, r1 = extract_tree_rules(X, y, feature_names=names, max_depth=3, seed=5)
    _, r2 = extract_tree_rules(X, y, feature_names=names, max_depth=3, seed=5)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2, strict=True):
        assert a.cluster == b.cluster
        assert a.conditions == b.conditions


def test_shap_attributions_returns_per_cluster_importance(blobs_with_names):
    X, y, names = blobs_with_names
    tree, _ = extract_tree_rules(
        X, y, feature_names=names, max_depth=3, min_samples_leaf=10, seed=0
    )
    attr = shap_attributions(tree, X, feature_names=names, background=50, seed=0)
    assert set(attr.keys()).issubset({0, 1, 2})
    for importances in attr.values():
        assert set(importances.keys()) == set(names)
        assert all(isinstance(v, float) for v in importances.values())


def test_shap_attributions_non_negative_mean_abs(blobs_with_names):
    X, y, names = blobs_with_names
    tree, _ = extract_tree_rules(X, y, feature_names=names, max_depth=3, seed=0)
    attr = shap_attributions(tree, X, feature_names=names, background=50)
    for importances in attr.values():
        assert all(v >= 0 for v in importances.values())


def test_shap_attributions_deterministic_for_fixed_seed(blobs_with_names):
    X, y, names = blobs_with_names
    tree, _ = extract_tree_rules(X, y, feature_names=names, max_depth=3, seed=0)
    a = shap_attributions(tree, X, feature_names=names, background=50, seed=7)
    b = shap_attributions(tree, X, feature_names=names, background=50, seed=7)
    for cls in a:
        for fn in a[cls]:
            assert a[cls][fn] == pytest.approx(b[cls][fn], rel=1e-5)
