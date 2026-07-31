"""Cluster interpretation (Phase 3, bonus +4).

Three complementary lenses on *why* a clusterer assigned points the way it did:

  - **decision-tree rule extraction**: fit a shallow DecisionTreeClassifier to
    predict cluster labels from the original features. Each root-to-leaf path
    is a human-readable rule characterising a cluster ("V14 < -2.3 and V4 > 1.1
    => cluster 3"). This is the most interpretable lens and works on any
    clusterer's output.

  - **SHAP feature attribution**: TreeExplainer on the surrogate tree gives
    per-feature, per-point Shapley values quantifying each feature's
    contribution to the cluster assignment. Aggregated, this yields a global
    feature-importance ranking per cluster.

  - **DiCE counterfactuals**: for a given point, find the minimal feature
    perturbation that flips its predicted cluster. This answers "what would
    have to change for this transaction to look like a different cluster?" —
    actionable for fraud analysis.

shap and dice_ml are imported lazily so the module imports without the [deep]
extra; only calling the relevant function requires the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree


@dataclass(frozen=True)
class TreeRule:
    cluster: int
    conditions: list[str]
    support: int
    precision: float


def extract_tree_rules(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    max_depth: int = 4,
    min_samples_leaf: int = 50,
    seed: int = 0,
) -> tuple[DecisionTreeClassifier, list[TreeRule]]:
    """Fit a surrogate decision tree and extract one rule per leaf.

    Returns (fitted_tree, rules). Each rule is the conjunction of inequalities
    on the path from root to leaf, labelled with the majority cluster at that
    leaf, its support, and precision (purity of the majority class).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    mask = labels != -1
    X, labels = X[mask], labels[mask]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    ).fit(X, labels)

    rules: list[TreeRule] = []
    t = tree.tree_
    feature = t.feature
    threshold = t.threshold
    classes = tree.classes_

    def recurse(node, conditions):
        if t.children_left[node] == _tree.TREE_LEAF:
            # ``tree_.value`` holds class *proportions* per node (sklearn >= 1.4)
            # and raw counts on older versions, so normalise before reading the
            # majority share, and take support from ``n_node_samples``.
            value = t.value[node].ravel()
            winner = int(np.argmax(value))
            # index into classes_, not the raw label: cluster ids need not be 0..k-1
            cluster = int(classes[winner])
            support = int(t.n_node_samples[node])
            total = float(value.sum())
            precision = float(value[winner] / total) if total else 0.0
            rules.append(TreeRule(cluster, list(conditions), support, precision))
            return
        f = feature_names[feature[node]]
        recurse(t.children_left[node], conditions + [f"{f} <= {threshold[node]:.4f}"])
        recurse(t.children_right[node], conditions + [f"{f} >  {threshold[node]:.4f}"])

    recurse(0, [])
    return tree, rules


def shap_attributions(
    tree: DecisionTreeClassifier,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    background: int = 200,
    seed: int = 0,
) -> dict:
    """Per-cluster SHAP feature importance from a surrogate tree.

    Returns ``{cluster_id: {feature_name: mean_abs_shap}}``. Uses
    TreeExplainer (exact for sklearn trees). Background subsample keeps the
    explainer tractable on DS-10.
    """
    import shap

    X = np.asarray(X)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    rng = np.random.default_rng(seed)
    if len(X) > background:
        bg_idx = rng.choice(len(X), size=background, replace=False)
        bg = X[bg_idx]
    else:
        bg = X
    explainer = shap.TreeExplainer(tree, data=bg)
    sv = explainer.shap_values(X)
    # shap >=0.42 returns ndarray (n, d, classes); older returns list[classes]
    if isinstance(sv, list):
        per_class = sv
    else:
        per_class = [sv[:, :, c] for c in range(sv.shape[2])] if sv.ndim == 3 else [sv]

    result = {}
    classes = getattr(tree, "classes_", list(range(len(per_class))))
    for ci, cls in enumerate(classes):
        arr = per_class[ci]
        mean_abs = np.abs(arr).mean(axis=0)
        result[int(cls)] = {fn: float(v) for fn, v in zip(feature_names, mean_abs, strict=True)}
    return result


def dice_counterfactuals(
    model,
    X: pd.DataFrame,
    query_index: int,
    target_cluster: int,
    feature_names: list[str],
    *,
    total_cfs: int = 3,
    features_to_vary: list[str] | None = None,
) -> pd.DataFrame:
    """Generate counterfactuals flipping ``query_index`` to ``target_cluster``.

    ``model`` must be a fitted sklearn classifier with ``predict``. Returns a
    DataFrame of counterfactual instances. Requires dice_ml (install via
    ``[deep]`` extra).
    """
    import dice_ml
    from dice_ml import Dice

    df = X.copy()
    continuous_features = list(df.select_dtypes(include=[np.number]).columns)
    data_interface = dice_ml.Data(
        dataframe=df,
        continuous_features=continuous_features,
        outcome_name="__cluster__",
    )
    model_interface = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    dice = Dice(data_interface, model_interface, method="random")
    query = df.iloc[[query_index]].copy()
    cf = dice.generate_counterfactuals(
        query,
        total_cfs=total_cfs,
        desired_class=int(target_cluster),
        features_to_vary=features_to_vary or feature_names,
    )
    return cf.cf_examples_df
