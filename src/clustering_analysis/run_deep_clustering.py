"""Pipeline: deep clustering, cluster interpretation, and deployment artefacts.

Entry point: ``python -m clustering_analysis.run_deep_clustering``
(``--tables-only`` re-emits the report tables without retraining).

This is the course's Phase 3 deliverable. It covers brief §4 end to end:

  §4.1 Track 1        AE+K-Means -> DEC -> DEC+InfoNCE, compared with the
                      Phase 2 winner on the *same* rows and metrics
  §4.2 interpretation cluster profiles, exemplars, surrogate-tree rules, SHAP
  §4.3 downstream     cluster-conditional modelling + anomaly ranking
  §4.4 fairness       amount / time-of-day composition + preprocessing ARI drift
  §4.5 production     model registry, schema validation, PSI/KS drift monitor
  bonus              bootstrap CIs + permutation tests on every headline metric

Deep models train on a stratified subsample (``phase3.train_n``): three networks
over 283,726 rows on CPU would dominate the pipeline's runtime for no
methodological gain, and the comparison against Phase 2 is run on identical rows
either way, which is what makes it fair.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import Params, load_params
from .dataset import load_labels, load_matrix, stratified_subsample
from .downstream import cluster_anomaly_detection, cluster_conditional_modelling
from .drift import drift_report, feature_drift, temporal_split
from .evaluation import PortfolioRun, agreement_matrix, performance_table, score_run
from .fairness import audit_cluster_composition, preprocessing_sensitivity, quantile_strata
from .interpretation import extract_tree_rules, shap_attributions
from .io_utils import interim_path, processed_path
from .metrics import ari
from .registry import data_version_from_manifest, register_model
from .reporting import (
    figure_path,
    results_path,
    safe_name,
    thousands,
    write_json,
    write_latex_table,
    write_matrix_table,
)
from .significance import bootstrap_ci, compare_models, permutation_test

REPO_ROOT = Path(__file__).resolve().parents[2]


def _feature_names(P: Params) -> list[str]:
    return list(P.scaling.v_features) + list(P.scaling.robust_features)


def _drift_time_axis(
    X_scaled: np.ndarray, sin_col: int, cos_col: int
) -> tuple[np.ndarray, str]:
    """Return the axis to split on for drift monitoring, and its provenance.

    The raw monotonic ``Time`` column is the correct split axis. Deriving the
    axis from ``time_sin``/``time_cos`` instead would be circular: those two
    features are inside the matrix being monitored, so splitting on them
    guarantees they are reported as the most drifted features regardless of
    whether anything actually drifted.

    Falls back to the cyclic phase only if the interim frame is unavailable, and
    reports which axis was used so the summary records the caveat.
    """
    cleaned = interim_path("cleaned.parquet")
    if cleaned.exists():
        import pandas as pd

        time_values = pd.read_parquet(cleaned, columns=["Time"])["Time"].to_numpy()
        if len(time_values) == len(X_scaled):
            return time_values, "raw Time column (monotonic)"
    return (
        np.arctan2(X_scaled[:, sin_col], X_scaled[:, cos_col]),
        "cyclic time phase (fallback: raw Time unavailable, so time_sin/time_cos "
        "drift is partly an artefact of the split axis)",
    )


def _as_run(name: str, labels: np.ndarray, runtime: float) -> PortfolioRun:
    unique = np.unique(labels)
    return PortfolioRun(
        algorithm=name,
        k=int(len(unique[unique != -1])),
        labels=labels,
        runtime_s=runtime,
        n_clusters=int(len(unique[unique != -1])),
        n_noise=int((labels == -1).sum()),
    )


# --------------------------------------------------------------------------- #
# §4.1 advanced track
# --------------------------------------------------------------------------- #
def run_deep_track(X: np.ndarray, k: int, P: Params) -> list[PortfolioRun]:
    """Train the three deep models and return them as comparable runs."""
    from .deep import fit_ae_kmeans, fit_dec, fit_dec_infonce

    ae = P.deep.autoencoder
    runs = []

    start = time.perf_counter()
    labels, _, _, _ = fit_ae_kmeans(
        X,
        k,
        latent_dim=ae.latent_dim,
        hidden_dims=list(ae.hidden_dims),
        ae_epochs=ae.epochs,
        ft_epochs=P.deep.ae_kmeans.epochs,
        batch_size=ae.batch_size,
        lr=ae.lr,
        seed=P.seeds.algorithms,
    )
    runs.append(_as_run("ae_kmeans", labels, time.perf_counter() - start))

    start = time.perf_counter()
    labels, _, _, dec_losses = fit_dec(
        X,
        k,
        latent_dim=ae.latent_dim,
        hidden_dims=list(ae.hidden_dims),
        init_epochs=P.deep.dec.init_epochs,
        dec_epochs=P.deep.dec.dec_epochs,
        update_interval=P.deep.dec.update_interval,
        tol=P.deep.dec.tol,
        gamma=P.deep.dec.gamma,
        batch_size=ae.batch_size,
        lr=ae.lr,
        seed=P.seeds.algorithms,
    )
    runs.append(_as_run("dec", labels, time.perf_counter() - start))

    start = time.perf_counter()
    labels, latent, _, ctn_losses = fit_dec_infonce(
        X,
        k,
        latent_dim=ae.latent_dim,
        hidden_dims=list(ae.hidden_dims),
        init_epochs=P.deep.dec.init_epochs,
        dec_epochs=P.deep.dec.dec_epochs,
        update_interval=P.deep.dec.update_interval,
        tol=P.deep.dec.tol,
        gamma=P.deep.dec.gamma,
        contrastive_weight=P.deep.dec_infonce.contrastive_weight,
        temperature=P.deep.dec_infonce.temperature,
        n_negatives=P.deep.dec_infonce.n_negatives,
        batch_size=ae.batch_size,
        lr=ae.lr,
        seed=P.seeds.algorithms,
    )
    runs.append(_as_run("dec_infonce", labels, time.perf_counter() - start))
    return runs, {"dec": dec_losses, "dec_infonce": ctn_losses}, latent


def plot_loss_curves(losses: dict[str, list[float]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, series in losses.items():
        ax.plot(series, label=f"{name} (final {series[-1]:.4f})", lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training objective")
    ax.set_title("Deep clustering objectives (brief §4.1.1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_latent_embedding(latent: np.ndarray, labels: np.ndarray, y: np.ndarray, path: Path) -> Path:
    """First two latent dimensions, coloured by cluster and by held-out Class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(latent[:, 0], latent[:, 1], c=labels, s=3, cmap="tab10", alpha=0.5)
    axes[0].set_title("DEC+InfoNCE latent space — cluster assignment")
    fraud = y == 1
    axes[1].scatter(latent[~fraud, 0], latent[~fraud, 1], s=2, alpha=0.15, label="legitimate")
    axes[1].scatter(latent[fraud, 0], latent[fraud, 1], s=14, color="red", label="fraud")
    axes[1].set_title("Same space — held-out Class overlay")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("latent-1")
        ax.set_ylabel("latent-2")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# §4.2 interpretation
# --------------------------------------------------------------------------- #
def cluster_profiles(
    features: np.ndarray,
    clusters: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    top_features: int = 3,
) -> list[dict]:
    """Per-cluster profile: size, fraud rate, and its most deviant features.

    Deviation is measured in population standard deviations from the global mean,
    which is what makes "this cluster is unusual in V14" a defensible statement
    rather than an eyeballed one.
    """
    global_mean = features.mean(axis=0)
    global_std = features.std(axis=0)
    global_std[global_std == 0] = 1.0

    rows = []
    for c in np.unique(clusters):
        mask = clusters == c
        z = (features[mask].mean(axis=0) - global_mean) / global_std
        order = np.argsort(-np.abs(z))[:top_features]
        rows.append(
            {
                "cluster": int(c),
                "size": int(mask.sum()),
                "share": float(mask.mean()),
                "fraud_rate": float(y[mask].mean()),
                "fraud_count": int(y[mask].sum()),
                "top_deviations": ", ".join(f"{feature_names[i]} ({z[i]:+.2f}σ)" for i in order),
            }
        )
    return sorted(rows, key=lambda r: -r["fraud_rate"])


def cluster_exemplars(
    features: np.ndarray,
    clusters: np.ndarray,
    *,
    sample_size: int = 10000,
    seed: int = 0,
) -> list[dict]:
    """Medoid, best-silhouette, and most marginal point per cluster (§4.2)."""
    from sklearn.metrics import silhouette_samples

    rng = np.random.default_rng(seed)
    pool = np.where(clusters != -1)[0]
    if len(pool) > sample_size:
        pool = np.sort(rng.choice(pool, size=sample_size, replace=False))
    sub_labels = clusters[pool]
    if len(np.unique(sub_labels)) < 2:
        return []
    sil = silhouette_samples(features[pool], sub_labels)

    rows = []
    for c in np.unique(sub_labels):
        local = np.where(sub_labels == c)[0]
        members = features[pool[local]]
        centroid = members.mean(axis=0)
        medoid_local = local[np.argmin(np.linalg.norm(members - centroid, axis=1))]
        best_local = local[np.argmax(sil[local])]
        worst_local = local[np.argmin(sil[local])]
        rows.append(
            {
                "cluster": int(c),
                "medoid_index": int(pool[medoid_local]),
                "best_silhouette_index": int(pool[best_local]),
                "best_silhouette": float(sil[best_local]),
                "boundary_index": int(pool[worst_local]),
                "boundary_silhouette": float(sil[worst_local]),
            }
        )
    return rows


def domain_labels(profiles: list[dict], base_rate: float) -> list[dict]:
    """Propose a human-readable name per cluster and justify it (§4.2).

    Naming is rule-based rather than hand-written so the label is reproducible
    and its justification is the same evidence a reader can check in the profile
    table. The thresholds are relative to the population fraud rate, so the
    naming survives a change of k or of algorithm.
    """
    named = []
    for p in profiles:
        enrichment = p["fraud_rate"] / base_rate if base_rate > 0 else float("nan")
        if p["fraud_count"] == 0:
            name, why = "Clean bulk", "no fraud at all; the benign core of the portfolio"
        elif enrichment >= 10:
            name, why = (
                "Fraud hotspot",
                f"fraud rate {enrichment:.0f}x the population rate ({p['fraud_rate']:.2%})",
            )
        elif enrichment >= 2:
            name, why = (
                "Elevated-risk pocket",
                f"fraud rate {enrichment:.1f}x the population rate",
            )
        elif p["share"] >= 0.4:
            name, why = "Mainstream traffic", f"{p['share']:.0%} of all transactions, near base rate"
        else:
            name, why = "Ordinary niche", "distinct geometry but unremarkable fraud rate"
        named.append(
            {
                "cluster": p["cluster"],
                "proposed_label": name,
                "size": p["size"],
                "fraud_rate": p["fraud_rate"],
                "enrichment_vs_base": float(enrichment),
                "justification": f"{why}; driven by {p['top_deviations']}",
            }
        )
    return named



# --------------------------------------------------------------------------- #
# report tables
# --------------------------------------------------------------------------- #
def write_tables(summary: dict, *, root: Path | None = None) -> None:
    """Emit every Phase 3 LaTeX table from the persisted summary alone.

    Separated from ``main`` for the same reason as in Phase 2: the report can be
    re-rendered after a wording change without retraining the deep models, and
    the tables are derived from ``summary.json`` so the two can never disagree.
    """
    tables = (root or REPO_ROOT) / "reports" / "tables"
    k = summary["selected_k"]
    n_sub = summary["subsample"]["n_rows"]

    write_latex_table(
        summary["track_comparison"],
        tables / "phase3_track_comparison.tex",
        caption="Advanced track vs the Phase 2 winner on the same "
        f"{thousands(n_sub)} stratified rows at $k={k}$.",
        label="tab:track",
    )

    write_matrix_table(
        summary["agreement"]["names"],
        np.asarray(summary["agreement"]["matrix"], dtype=float),
        tables / "phase3_agreement.tex",
        caption="Pairwise ARI: does the deep track find the Phase 2 structure or a different one?",
        label="tab:phase3-agreement",
    )

    write_latex_table(
        summary["significance"]["per_clustering"],
        tables / "phase3_significance.tex",
        caption="Bootstrap confidence intervals and permutation p-values for ARI "
        "against the held-out Class label.",
        label="tab:significance",
        precision=4,
    )

    write_latex_table(
        summary["profiles"],
        tables / "phase3_profiles.tex",
        columns=["cluster", "size", "share", "fraud_rate", "fraud_count", "top_deviations"],
        caption="Cluster profiles for the preferred Phase 3 clustering "
        f"({safe_name(summary['best_deep'])}).",
        label="tab:profiles",
        precision=4,
    )

    write_latex_table(
        summary["exemplars"],
        tables / "phase3_exemplars.tex",
        caption="Per-cluster medoid, highest-silhouette, and boundary exemplars.",
        label="tab:exemplars",
    )

    write_latex_table(
        summary["domain_labels"],
        tables / "phase3_domain_labels.tex",
        columns=[
            "cluster",
            "proposed_label",
            "size",
            "fraud_rate",
            "enrichment_vs_base",
            "justification",
        ],
        caption="Proposed domain labels with the profile evidence behind each name.",
        label="tab:domain-labels",
        precision=4,
    )

    tree_info = summary["surrogate_tree"]
    write_latex_table(
        tree_info["rules"],
        tables / "phase3_rules.tex",
        columns=["cluster", "support", "precision", "rule"],
        caption=f"Surrogate decision-tree rules (fidelity "
        f"{tree_info['accuracy'] * 100:.1f}\\%), "
        "ten highest-support leaves.",
        label="tab:rules",
    )

    write_latex_table(
        summary["shap_rows"],
        tables / "phase3_shap.tex",
        caption="Top SHAP attributions per cluster from the surrogate tree.",
        label="tab:shap",
    )

    cond = summary["downstream"]["conditional_modelling"]
    write_latex_table(
        [
            {
                "model": "global (clustering ignored)",
                "average_precision": cond["global_average_precision"],
                "roc_auc": cond["global_roc_auc"],
            },
            {
                "model": "cluster-conditional",
                "average_precision": cond["clustered_average_precision"],
                "roc_auc": cond["clustered_roc_auc"],
            },
        ],
        tables / "phase3_downstream.tex",
        caption="Cluster-conditional vs global modelling on a stratified holdout "
        f"({thousands(cond['n_test'])} rows).",
        label="tab:downstream",
        precision=4,
    )

    anomaly = summary["downstream"]["anomaly_detection"]
    write_latex_table(
        [
            {
                "quantity": f"transactions flagged (top {anomaly['top_fraction']:.0%})",
                "value": float(anomaly["n_flagged"]),
            },
            {"quantity": "precision@k", "value": anomaly["precision_at_k"]},
            {"quantity": "recall@k", "value": anomaly["recall_at_k"]},
            {"quantity": "population base rate", "value": anomaly["base_rate"]},
            {"quantity": "enrichment vs random review", "value": anomaly["enrichment"]},
            {"quantity": "ROC-AUC of the distance ranking", "value": anomaly["roc_auc"]},
            {"quantity": "average precision of the ranking", "value": anomaly["average_precision"]},
        ],
        tables / "phase3_anomaly.tex",
        caption="Cluster-based anomaly detection: ranking by distance to the assigned centroid.",
        label="tab:anomaly",
        precision=4,
    )

    write_latex_table(
        [
            {
                "attribute": a["attribute"],
                "strata": a["n_strata"],
                "max_representation_ratio": a["max_representation_ratio"],
                "clusters_flagged": len(a["concentrated_clusters"]),
            }
            for a in summary["fairness"]
        ],
        tables / "phase3_fairness.tex",
        caption="Proxy-attribute audit: strongest stratum over-representation per attribute "
        "(ratio 1.0 = the cluster mirrors the population).",
        label="tab:fairness",
    )

    write_latex_table(
        summary["sensitivity"],
        tables / "phase3_sensitivity.tex",
        columns=["variant", "n_clusters", "ari_vs_reference", "description"],
        caption="Preprocessing sensitivity: ARI of a re-clustering under each variant "
        "against the preferred clustering.",
        label="tab:sensitivity",
    )

    write_latex_table(
        summary["drift"]["top_features"],
        tables / "phase3_drift.tex",
        caption="Drift between the early and late halves of the time axis; the re-fit "
        f"trigger is PSI $\\geq$ {summary['drift']['summary']['psi_threshold']}.",
        label="tab:drift",
        precision=4,
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(params_path: Path | str = REPO_ROOT / "params.yaml") -> dict:
    P = load_params(params_path)
    started = time.perf_counter()
    seed = P.seeds.algorithms
    names = _feature_names(P)

    X_deep = load_matrix(P.phase3.input_matrix)
    y_full = load_labels()
    # "phase2_clusters.npy" is the artefact name the portfolio pipeline writes;
    # it holds that pipeline's preferred full-data labelling.
    portfolio_labels = np.load(processed_path("phase2_clusters.npy"))
    portfolio_summary_path = results_path("phase2", "summary.json")
    if not portfolio_summary_path.exists():
        raise FileNotFoundError(
            "results/phase2/summary.json missing — run "
            "`python -m clustering_analysis.run_portfolio` first"
        )
    portfolio = json.loads(portfolio_summary_path.read_text())
    k = int(portfolio["selected_k"])

    idx = stratified_subsample(y_full, P.phase3.train_n, seed=seed)
    X_sub, y_sub, portfolio_sub = X_deep[idx], y_full[idx], portfolio_labels[idx]

    summary: dict = {
        "selected_k": k,
        "phase2_preferred": portfolio["preferred"]["algorithm"],
        "n_rows_full": int(len(X_deep)),
        "subsample": {"n_rows": int(len(idx)), "n_fraud": int(y_sub.sum())},
        "feature_space": P.phase3.input_matrix,
    }

    # ---- §4.1 deep track vs Phase 2 winner --------------------------------- #
    print(f"[phase3] deep track on {len(X_sub)} rows, k={k} ...", flush=True)
    deep_runs, losses, latent = run_deep_track(X_sub, k, P)

    baseline = _as_run(
        f"phase2:{portfolio['preferred']['algorithm']}", portfolio_sub, float("nan")
    )
    all_runs = [baseline, *deep_runs]
    for run in all_runs:
        score_run(
            run,
            X_sub,
            y_sub,
            internal_metrics=P.metrics.internal,
            external_metrics=P.metrics.external,
            silhouette_sample_size=P.k_selection.silhouette.sample_size,
            seed=seed,
        )
    summary["track_comparison"] = performance_table(all_runs)
    plot_loss_curves(losses, figure_path("phase3_deep_losses.pdf"))
    plot_latent_embedding(latent, deep_runs[-1].labels, y_sub, figure_path("phase3_latent.pdf"))

    agree_names, M = agreement_matrix(all_runs)
    summary["agreement"] = {"names": agree_names, "matrix": M}

    best_deep = max(
        deep_runs,
        key=lambda r: (-1e9 if np.isnan(r.internal.get("silhouette", np.nan)) else r.internal["silhouette"]),
    )
    summary["best_deep"] = best_deep.algorithm

    # ---- bonus: significance ----------------------------------------------- #
    print("[phase3] significance testing ...", flush=True)
    sig_rows = []
    for run in all_runs:
        ci = bootstrap_ci(
            ari,
            y_sub,
            run.labels,
            n_bootstrap=min(P.significance.n_bootstrap, 300),
            level=P.significance.ci_level,
            seed=P.significance.permutation_seed,
        )
        perm = permutation_test(
            ari,
            y_sub,
            run.labels,
            n_permutations=min(P.significance.n_permutations, 300),
            seed=P.significance.permutation_seed,
        )
        sig_rows.append(
            {
                "clustering": run.algorithm,
                "ari": ci.observed,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "null_mean": perm.null_mean,
                "p_value": perm.p_value,
            }
        )
    head_to_head = compare_models(
        ari,
        y_sub,
        best_deep.labels,
        baseline.labels,
        n_bootstrap=min(P.significance.n_bootstrap, 300),
        level=P.significance.ci_level,
        seed=P.significance.permutation_seed,
    )
    summary["significance"] = {"per_clustering": sig_rows, "best_deep_vs_phase2": head_to_head}

    # ---- §4.2 interpretation ----------------------------------------------- #
    print("[phase3] interpretation ...", flush=True)
    features = np.load(processed_path("X_scaled.npy"))[idx]
    chosen = best_deep.labels
    profiles = cluster_profiles(features, chosen, y_sub, names)
    summary["profiles"] = profiles

    exemplars = cluster_exemplars(features, chosen, seed=seed)
    summary["exemplars"] = exemplars

    labelled = domain_labels(profiles, float(y_full.mean()))
    summary["domain_labels"] = labelled

    tree, rules = extract_tree_rules(
        features,
        chosen,
        names,
        max_depth=P.interpretation.tree_max_depth,
        min_samples_leaf=P.interpretation.tree_min_samples_leaf,
        seed=seed,
    )
    accuracy = float(tree.score(features, chosen))
    top_rules = sorted(rules, key=lambda r: -r.support)[:10]
    summary["surrogate_tree"] = {
        "accuracy": accuracy,
        "n_rules": len(rules),
        "max_depth": P.interpretation.tree_max_depth,
        "rules": [
            {
                "cluster": r.cluster,
                "support": r.support,
                "precision": r.precision,
                "rule": " AND ".join(r.conditions) or "(root)",
            }
            for r in top_rules
        ],
    }

    shap_top: dict = {}
    shap_error = None
    try:
        attributions = shap_attributions(
            tree, features, names, background=P.interpretation.shap_background, seed=seed
        )
        for cluster_id, importances in attributions.items():
            ranked = sorted(importances.items(), key=lambda kv: -kv[1])[:5]
            shap_top[int(cluster_id)] = [{"feature": f, "mean_abs_shap": v} for f, v in ranked]
    except Exception as exc:  # pragma: no cover - depends on the optional shap build
        shap_error = str(exc)
    summary["shap_top_features"] = shap_top if shap_error is None else {"error": shap_error}
    # The table is written unconditionally: the report \inputs it, so skipping it
    # on a SHAP failure would break the LaTeX build rather than degrade one row.
    summary["shap_rows"] = (
        [
            {
                "cluster": cid,
                "top_features": ", ".join(
                    f"{e['feature']} ({e['mean_abs_shap']:.3f})" for e in entries
                ),
            }
            for cid, entries in sorted(shap_top.items())
        ]
        if shap_error is None
        else [{"cluster": "n/a", "top_features": f"SHAP unavailable: {shap_error}"}]
    )

    # ---- §4.3 downstream --------------------------------------------------- #
    print("[phase3] downstream analysis ...", flush=True)
    cond = cluster_conditional_modelling(
        features,
        y_sub,
        chosen,
        test_fraction=P.phase3.downstream.test_fraction,
        seed=seed,
    )
    anomaly = cluster_anomaly_detection(features, chosen, y_sub, top_fraction=0.01)
    summary["downstream"] = {
        "conditional_modelling": {
            "n_train": cond.n_train,
            "n_test": cond.n_test,
            "global_average_precision": cond.global_average_precision,
            "clustered_average_precision": cond.clustered_average_precision,
            "global_roc_auc": cond.global_roc_auc,
            "clustered_roc_auc": cond.clustered_roc_auc,
            "lift": cond.lift,
            "per_cluster": cond.per_cluster,
        },
        "anomaly_detection": {
            "n_flagged": anomaly.n_flagged,
            "top_fraction": anomaly.top_fraction,
            "precision_at_k": anomaly.precision_at_k,
            "recall_at_k": anomaly.recall_at_k,
            "base_rate": anomaly.base_rate,
            "enrichment": anomaly.enrichment,
            "roc_auc": anomaly.roc_auc,
            "average_precision": anomaly.average_precision,
        },
    }
    # ---- §4.4 fairness + sensitivity --------------------------------------- #
    print("[phase3] fairness + sensitivity audit ...", flush=True)
    amount_col = names.index("log_amount")
    audits = []
    amount_audit = audit_cluster_composition(
        chosen, quantile_strata(features[:, amount_col], n_strata=4), attribute="log_amount quartile"
    )
    audits.append(amount_audit)
    sin_col, cos_col = names.index("time_sin"), names.index("time_cos")
    phase = np.arctan2(features[:, sin_col], features[:, cos_col])
    time_audit = audit_cluster_composition(
        chosen, quantile_strata(phase, n_strata=4), attribute="time-of-day quartile"
    )
    audits.append(time_audit)
    summary["fairness"] = [
        {
            "attribute": a.attribute,
            "n_strata": a.n_strata,
            "max_representation_ratio": a.max_representation_ratio,
            "concentrated_clusters": a.concentrated_clusters,
            "per_cluster": a.per_cluster,
        }
        for a in audits
    ]

    variants = {"pca_space": (load_matrix("X_pca")[idx], "PCA at 95% variance instead of scaled features")}
    try:
        from .scaling import build_scaler

        raw_features = interim_path("features.parquet")
        if raw_features.exists():
            import pandas as pd

            df = pd.read_parquet(raw_features).iloc[idx]
            alt = build_scaler(
                P.phase3.sensitivity.alt_scaler,
                v_features=list(P.scaling.v_features),
                robust_features=list(P.scaling.robust_features),
            ).fit_transform(df)
            variants[f"{P.phase3.sensitivity.alt_scaler}_scaler"] = (
                alt,
                f"uniform {P.phase3.sensitivity.alt_scaler} scaling instead of the hybrid choice",
            )
    except Exception as exc:  # pragma: no cover - depends on interim artefacts
        summary.setdefault("warnings", []).append(f"alt-scaler variant skipped: {exc}")

    sens = preprocessing_sensitivity(variants, chosen, k, seed=seed)
    summary["sensitivity"] = [
        {
            "variant": s.variant,
            "n_clusters": s.n_clusters,
            "ari_vs_reference": s.ari_vs_reference,
            "description": s.description,
        }
        for s in sens
    ]

    # ---- §4.5 production: registry + drift ---------------------------------- #
    print("[phase3] registry + drift monitoring ...", flush=True)
    X_scaled_full = np.load(processed_path("X_scaled.npy"))
    time_axis, time_axis_source = _drift_time_axis(X_scaled_full, sin_col, cos_col)
    early, late = temporal_split(
        X_scaled_full, time_axis, split_fraction=P.phase3.drift.split_fraction
    )
    drifts = feature_drift(
        early,
        late,
        names,
        n_bins=P.phase3.drift.n_bins,
        psi_threshold=P.phase3.drift.psi_threshold,
    )
    report = drift_report(drifts, psi_threshold=P.phase3.drift.psi_threshold)
    summary["drift"] = {
        "split_axis": time_axis_source,
        "summary": report,
        "top_features": [
            {"feature": d.feature, "psi": d.psi, "ks_statistic": d.ks_statistic, "ks_pvalue": d.ks_pvalue, "drifted": d.drifted}
            for d in drifts[:10]
        ],
    }

    card = register_model(
        name="phase3-preferred",
        algorithm=best_deep.algorithm,
        k=k,
        data_version=data_version_from_manifest(),
        n_rows=int(len(idx)),
        n_features=int(X_sub.shape[1]),
        feature_space=P.phase3.input_matrix,
        params={
            "latent_dim": P.deep.autoencoder.latent_dim,
            "hidden_dims": list(P.deep.autoencoder.hidden_dims),
            "dec_epochs": P.deep.dec.dec_epochs,
            "contrastive_weight": P.deep.dec_infonce.contrastive_weight,
            "temperature": P.deep.dec_infonce.temperature,
        },
        metrics={**best_deep.internal, **best_deep.external},
        seeds={"algorithms": seed, "global": P.seeds.global_},
        notes=(
            f"Phase 3 Track 1 winner; compared against Phase 2 "
            f"{portfolio['preferred']['algorithm']} on identical rows."
        ),
        tags=["phase3", "deep-clustering", "track1"],
    )
    summary["registry_entry"] = {"entry_id": card.entry_id, "fit_date": card.fit_date}

    np.save(processed_path("phase3_clusters.npy").as_posix(), chosen)
    np.save(processed_path("phase3_subsample_idx.npy").as_posix(), idx)
    np.save(processed_path("phase3_latent.npy").as_posix(), latent)

    summary["runtime_s"] = round(time.perf_counter() - started, 1)
    write_json(summary, results_path("phase3", "summary.json"))
    write_tables(summary)
    print(f"[phase3] done in {summary['runtime_s']}s", flush=True)
    return summary


def refresh_tables() -> dict:
    """Re-emit the tables from the last run's persisted summary."""
    path = results_path("phase3", "summary.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the phase3 stage first")
    summary = json.loads(path.read_text())
    write_tables(summary)
    print(f"[phase3] tables refreshed from {path}", flush=True)
    return summary


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        refresh_tables()
    else:
        main()
