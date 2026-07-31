"""Pipeline: compare a portfolio of classical clustering algorithms.

Entry point: ``python -m clustering_analysis.run_portfolio``
(``--tables-only`` re-emits the report tables without re-clustering).

This is the course's Phase 2 deliverable. It runs the whole of brief §3 end to
end and writes every number it reports:

  §3.1 portfolio      four families (K-Means, Ward, HDBSCAN, GMM)
  §3.2 determining k  elbow, silhouette, gap, BIC, bootstrap stability
  §3.3 internal       silhouette, Davies-Bouldin, Calinski-Harabasz, Dunn
  §3.4 external       ARI, NMI, AMI, V-measure, purity (Class as ground truth)
  §3.5 stability       seed ARI + co-association consensus
  §3.6 comparison     performance table, pairwise ARI, error analysis
  §2.6 metric ablation clustering under Euclidean / Manhattan / cosine

**Two evaluation regimes, reported separately.** K-Means, GMM and HDBSCAN run on
all 283,726 rows. Ward cannot: its pairwise distance matrix would be ~322 GB. So
a second table re-runs all four families on one shared stratified subsample,
which is what makes the four-family comparison like-for-like. Mixing the two
into a single table would compare a full-data score against a subsample score,
so the driver keeps them apart and the report states which is which.

Labels (``Class``) are never an input to any fit — only to the external metrics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .algorithms.gmm import compare_covariance_types
from .config import Params, load_params
from .dataset import load_labels, load_matrix, stratified_subsample
from .evaluation import (
    PortfolioRun,
    agreement_matrix,
    error_analysis,
    metric_ablation,
    performance_table,
    rank_runs,
    run_algorithm,
    score_run,
)
from .hierarchical import (
    build_linkages,
    cophenetic_correlation,
    cut_at_height_fraction,
    cut_by_max_silhouette,
    linkage_comparison,
)
from .io_utils import processed_path
from .k_selection import (
    aggregate_k_selections,
    select_k_bic,
    select_k_bootstrap,
    select_k_elbow,
    select_k_gap,
    select_k_silhouette,
)
from .reporting import (
    figure_path,
    results_path,
    safe_name,
    thousands,
    write_json,
    write_latex_table,
    write_matrix_table,
)
from .stability import co_association_matrix, consensus_clustering, seed_stability

REPO_ROOT = Path(__file__).resolve().parents[2]


def _family_kwargs(P: Params, name: str) -> dict:
    fams = P.algorithms.families
    return {
        "kmeans": {"n_init": fams.kmeans.n_init},
        "ward": {"linkage": fams.ward.linkage},
        "gmm": {"covariance_type": fams.gmm.covariance_type},
        "hdbscan": {
            "min_cluster_size": fams.hdbscan.min_cluster_size,
            "min_samples": fams.hdbscan.min_samples,
            "cluster_selection_method": fams.hdbscan.cluster_selection_method,
        },
    }[name]


# --------------------------------------------------------------------------- #
# §3.2 determining k
# --------------------------------------------------------------------------- #
def run_k_selection(X: np.ndarray, P: Params) -> dict:
    """Five independent criteria for k, plus the vote across them."""
    k_range = P.algorithms.k_range
    seed = P.seeds.algorithms
    results = [
        select_k_elbow(X, k_range, seed=seed),
        select_k_silhouette(
            X, k_range, seed=seed, sample_size=P.k_selection.silhouette.sample_size
        ),
        select_k_gap(X, k_range, n_refs=P.k_selection.gap.n_refs, seed=seed),
        select_k_bic(X, k_range, seed=seed, covariance_type=P.algorithms.families.gmm.covariance_type),
        select_k_bootstrap(
            X,
            k_range,
            n_resamples=P.k_selection.bootstrap.n_resamples,
            sample_fraction=P.k_selection.bootstrap.sample_fraction,
            seed=seed,
        ),
    ]
    votes = aggregate_k_selections(results)
    # Break ties on the primary internal metric's own recommendation, which is
    # the criterion the report defends its final k with.
    silhouette_k = next(r.recommended_k for r in results if r.method == "silhouette")
    best = max(votes.items(), key=lambda kv: (kv[1], kv[0] == silhouette_k))
    return {
        "k_range": k_range,
        "methods": [
            {
                "method": r.method,
                "recommended_k": r.recommended_k,
                "higher_is_better": r.higher_is_better,
                "scores": {str(k): v for k, v in r.scores.items()},
            }
            for r in results
        ],
        "votes": {str(k): v for k, v in votes.items()},
        "consensus_k": int(best[0]),
        "n_rows": int(len(X)),
    }


def plot_k_selection(k_selection: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = k_selection["methods"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.4))
    for ax, m in zip(np.atleast_1d(axes), methods, strict=True):
        ks = sorted(int(k) for k in m["scores"])
        ys = [m["scores"][str(k)] for k in ks]
        ax.plot(ks, ys, marker="o", lw=1.4)
        ax.axvline(m["recommended_k"], color="crimson", ls="--", lw=1.2)
        ax.set_title(f"{m['method']} → k={m['recommended_k']}", fontsize=10)
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)
    fig.suptitle("Determining k — five independent criteria (brief §3.2)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# §3.1 + §3.3 + §3.4 portfolio
# --------------------------------------------------------------------------- #
def run_portfolio(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    P: Params,
    *,
    algorithms: list[str],
) -> list[PortfolioRun]:
    runs = []
    for name in algorithms:
        run = run_algorithm(name, X, k, seed=P.seeds.algorithms, **_family_kwargs(P, name))
        runs.append(
            score_run(
                run,
                X,
                y,
                internal_metrics=P.metrics.internal,
                external_metrics=P.metrics.external,
                silhouette_sample_size=P.k_selection.silhouette.sample_size,
                seed=P.seeds.algorithms,
            )
        )
    return runs


def plot_agreement(names: list[str], M: np.ndarray, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(names), 1.2 + 1.0 * len(names)))
    im = ax.imshow(M, cmap="viridis", vmin=-0.05, vmax=1.0)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if M[i, j] < 0.6 else "black",
            )
    fig.colorbar(im, label="ARI")
    ax.set_title("Algorithm-pair agreement (ARI)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_consensus(M: np.ndarray, consensus: np.ndarray, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(consensus, kind="stable")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, label="co-clustering probability")
    ax.set_title("Co-association matrix, ordered by consensus label")
    ax.set_xlabel("point (reordered)")
    ax.set_ylabel("point (reordered)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dendrograms(trees: dict[str, np.ndarray], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, axes = plt.subplots(1, len(trees), figsize=(4.2 * len(trees), 3.6))
    for ax, (name, Z) in zip(np.atleast_1d(axes), trees.items(), strict=True):
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, no_labels=True, color_threshold=0)
        ax.set_title(f"{name} linkage", fontsize=10)
        ax.set_ylabel("merge height")
    fig.suptitle("Dendrogram topology by linkage criterion (brief §3.1.2)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_silhouette_distribution(X: np.ndarray, labels: np.ndarray, path: Path, *, seed: int = 0) -> Path:
    """Per-point silhouette distribution, not just its mean (brief §3.2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_samples

    rng = np.random.default_rng(seed)
    mask = np.where(labels != -1)[0]
    if len(mask) > 20000:
        mask = np.sort(rng.choice(mask, size=20000, replace=False))
    sil = silhouette_samples(X[mask], labels[mask])
    sub = labels[mask]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_lower = 0
    for c in np.unique(sub):
        vals = np.sort(sil[sub == c])
        ax.fill_betweenx(np.arange(y_lower, y_lower + len(vals)), 0, vals, alpha=0.8)
        ax.text(-0.05, y_lower + len(vals) / 2, str(int(c)), fontsize=8, va="center")
        y_lower += len(vals) + 20
    ax.axvline(sil.mean(), color="crimson", ls="--", label=f"mean = {sil.mean():.3f}")
    ax.set_xlabel("silhouette coefficient")
    ax.set_ylabel("points, grouped by cluster")
    ax.set_title("Per-point silhouette distribution — preferred clustering")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# report tables
# --------------------------------------------------------------------------- #
def write_tables(summary: dict, *, root: Path | None = None) -> None:
    """Emit every Phase 2 LaTeX table from the persisted summary alone.

    Kept separate from ``main`` so the report can be regenerated after a wording
    or formatting change without re-running ~40 minutes of clustering:
    ``python -m clustering_analysis.run_portfolio --tables-only``. It also guarantees
    that the tables and ``summary.json`` can never disagree, since the tables are
    derived from the summary rather than from live variables.
    """
    tables = (root or REPO_ROOT) / "reports" / "tables"
    k = summary["selected_k"]
    n_ks = summary["k_selection"]["n_rows"]
    n_sub = summary["subsample"]["n_rows"]

    write_latex_table(
        [
            {
                "method": m["method"],
                "recommended_k": m["recommended_k"],
                "direction": "max" if m["higher_is_better"] else "min",
                "score_at_recommended": m["scores"][str(m["recommended_k"])],
            }
            for m in summary["k_selection"]["methods"]
        ],
        tables / "phase2_k_selection.tex",
        caption=f"Determining $k$: five criteria on {thousands(n_ks)} stratified rows. "
        f"Consensus $k={k}$.",
        label="tab:k-selection",
    )

    write_latex_table(
        summary["portfolio_full"],
        tables / "phase2_portfolio_full.tex",
        caption=f"Portfolio on all {thousands(summary['n_rows_full'])} rows at $k={k}$ "
        "(Ward excluded: its distance matrix is $\\sim$322\\,GB).",
        label="tab:portfolio-full",
    )

    write_latex_table(
        summary["portfolio_subsample"],
        tables / "phase2_portfolio_subsample.tex",
        caption="All four families on one shared stratified subsample "
        f"({thousands(n_sub)} rows, {summary['subsample']['n_fraud']} fraud) at $k={k}$.",
        label="tab:portfolio-sub",
    )

    write_matrix_table(
        summary["agreement"]["names"],
        np.asarray(summary["agreement"]["matrix"], dtype=float),
        tables / "phase2_agreement.tex",
        caption="Pairwise ARI between the four families on the shared subsample.",
        label="tab:agreement",
    )

    write_latex_table(
        summary["linkage_comparison"],
        tables / "phase2_linkage.tex",
        caption="All four linkage criteria on one distance matrix, with cophenetic "
        "correlation and the share of points absorbed by the largest cluster.",
        label="tab:linkage",
    )

    cuts = summary["dendrogram_cuts"]
    write_latex_table(
        [
            {
                "strategy": "fixed height (0.7 of max merge)",
                "n_clusters": cuts["fixed_height"]["n_clusters"],
                "criterion_value": cuts["fixed_height"]["height"],
                "ari_vs_class": cuts["fixed_height"]["ari_vs_class"],
            },
            {
                "strategy": "max silhouette over k",
                "n_clusters": cuts["max_silhouette"]["k"],
                "criterion_value": cuts["max_silhouette"]["silhouette"],
                "ari_vs_class": cuts["max_silhouette"]["ari_vs_class"],
            },
        ],
        tables / "phase2_cuts.tex",
        caption="Two dendrogram cutting strategies on the Ward tree.",
        label="tab:cuts",
    )

    write_latex_table(
        summary["gmm_covariance"],
        tables / "phase2_gmm_covariance.tex",
        columns=["covariance_type", "mean_log_likelihood", "bic", "aic", "n_parameters", "n_iter"],
        caption="GMM covariance structures at the selected $k$, sorted by BIC.",
        label="tab:gmm-cov",
        precision=1,
    )

    write_latex_table(
        summary["metric_ablation"],
        tables / "phase2_metric_ablation.tex",
        caption="Average-linkage clustering under three distance metrics; "
        "ARI is measured against the Euclidean reference.",
        label="tab:ablation",
    )

    write_latex_table(
        summary["consensus"]["comparison"],
        tables / "phase2_consensus.tex",
        caption="Consensus clustering vs its best single base clusterer on the "
        "same rows (brief §3.5).",
        label="tab:consensus",
    )

    ea = summary["error_analysis"]
    preferred = summary["preferred"]
    write_latex_table(
        [
            {"quantity": "points inspected (lowest silhouette)", "value": float(ea["n_inspected"])},
            {"quantity": "silhouette cut-off", "value": ea["silhouette_threshold"]},
            {"quantity": "mean silhouette (all points)", "value": ea["mean_silhouette"]},
            {"quantity": "fraction with negative silhouette", "value": ea["frac_negative_silhouette"]},
            {"quantity": "fraud rate among inspected", "value": ea["fraud_rate_worst"]},
            {"quantity": "fraud rate overall", "value": ea["fraud_rate_overall"]},
            {"quantity": "fraud enrichment factor", "value": ea["fraud_enrichment"]},
        ],
        tables / "phase2_error_analysis.tex",
        caption="Error analysis of the preferred clustering "
        f"({safe_name(preferred['algorithm'])}, $k={preferred['k']}$).",
        label="tab:error-analysis",
        precision=5,
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(params_path: Path | str = REPO_ROOT / "params.yaml") -> dict:
    P = load_params(params_path)
    started = time.perf_counter()

    X_full = load_matrix(P.phase2.input_matrix)
    y_full = load_labels()
    if len(X_full) != len(y_full):
        raise ValueError(f"Feature/label length mismatch: {len(X_full)} vs {len(y_full)}")

    sub_idx = stratified_subsample(y_full, P.phase2.portfolio_n, seed=P.seeds.algorithms)
    X_sub, y_sub = X_full[sub_idx], y_full[sub_idx]

    ks_idx = stratified_subsample(y_full, P.phase2.k_selection_n, seed=P.seeds.algorithms + 1)
    X_ks = X_full[ks_idx]

    summary: dict = {
        "input_matrix": P.phase2.input_matrix,
        "n_rows_full": int(len(X_full)),
        "n_features": int(X_full.shape[1]),
        "fraud_rate_full": float(y_full.mean()),
        "subsample": {
            "n_rows": int(len(X_sub)),
            "fraud_rate": float(y_sub.mean()),
            "n_fraud": int(y_sub.sum()),
        },
    }

    # ---- §3.2 determining k ------------------------------------------------ #
    print(f"[phase2] k-selection on {len(X_ks)} rows ...", flush=True)
    k_selection = run_k_selection(X_ks, P)
    k = k_selection["consensus_k"]
    summary["k_selection"] = k_selection
    summary["selected_k"] = k
    plot_k_selection(k_selection, figure_path("phase2_k_selection.pdf"))

    # ---- §3.1/§3.3/§3.4 portfolio, full data ------------------------------- #
    print(f"[phase2] full-data portfolio at k={k} ...", flush=True)
    full_runs = run_portfolio(X_full, y_full, k, P, algorithms=list(P.phase2.full_algorithms))
    summary["portfolio_full"] = performance_table(full_runs)

    # ---- four-family comparison on shared rows ----------------------------- #
    print(f"[phase2] four-family portfolio on {len(X_sub)} shared rows ...", flush=True)
    sub_runs = run_portfolio(X_sub, y_sub, k, P, algorithms=["kmeans", "ward", "gmm", "hdbscan"])
    summary["portfolio_subsample"] = performance_table(sub_runs)

    # ---- §3.6 agreement ---------------------------------------------------- #
    names, M_agree = agreement_matrix(sub_runs)
    summary["agreement"] = {"names": names, "matrix": M_agree}
    plot_agreement(names, M_agree, figure_path("phase2_agreement.pdf"))

    # ---- §3.1.2 Group B: linkages, cophenetic, two cuts -------------------- #
    print("[phase2] linkage comparison + cophenetic correlation ...", flush=True)
    # one O(n²) distance vector, reused by every linkage routine below
    trees, distances = build_linkages(X_sub)
    diag = linkage_comparison(X_sub, k, trees=trees, distances=distances)
    summary["linkage_comparison"] = [
        {
            "linkage": d.linkage,
            "cophenetic_corr": d.cophenetic_corr,
            "n_clusters": d.n_clusters,
            "max_merge_height": d.max_merge_height,
            "largest_cluster_frac": d.cluster_sizes[0] / len(X_sub),
        }
        for d in diag
    ]

    plot_dendrograms(trees, figure_path("phase2_dendrograms.pdf"))
    Z_ward = trees["ward"]
    height_labels, height = cut_at_height_fraction(Z_ward, fraction=0.7)
    sil_labels, sil_k, sil_scores = cut_by_max_silhouette(
        Z_ward, X_sub, P.algorithms.k_range, sample_size=P.k_selection.silhouette.sample_size
    )
    from sklearn.metrics import adjusted_rand_score

    summary["dendrogram_cuts"] = {
        "ward_cophenetic": cophenetic_correlation(Z_ward, distances),
        "fixed_height": {
            "height": height,
            "n_clusters": int(len(np.unique(height_labels))),
            "ari_vs_class": float(adjusted_rand_score(y_sub, height_labels)),
        },
        "max_silhouette": {
            "k": sil_k,
            "silhouette": sil_scores[sil_k],
            "ari_vs_class": float(adjusted_rand_score(y_sub, sil_labels)),
        },
        "ari_between_cuts": float(adjusted_rand_score(height_labels, sil_labels)),
    }

    # ---- §3.1.4 Group D: covariance structures ----------------------------- #
    print("[phase2] GMM covariance comparison ...", flush=True)
    cov_rows = compare_covariance_types(X_sub, k, seed=P.seeds.algorithms)
    summary["gmm_covariance"] = cov_rows

    # ---- §2.6 distance-metric ablation ------------------------------------- #
    print("[phase2] distance-metric ablation ...", flush=True)
    abl_idx = stratified_subsample(y_full, P.phase2.ablation_n, seed=P.seeds.algorithms + 2)
    abl_rows = metric_ablation(X_full[abl_idx], k)
    summary["metric_ablation"] = abl_rows

    # ---- §3.5 stability + consensus ---------------------------------------- #
    print("[phase2] seed stability ...", flush=True)
    stability = {}
    for name in ("kmeans", "gmm"):
        mean_ari, _ = seed_stability(
            name,
            X_sub,
            k,
            n_seeds=P.stability.n_seeds,
            seed_offset=P.stability.seed_offset,
            **_family_kwargs(P, name),
        )
        stability[name] = {"n_seeds": P.stability.n_seeds, "mean_pairwise_ari": mean_ari}
    summary["seed_stability"] = stability

    print("[phase2] co-association consensus ...", flush=True)
    M_co = co_association_matrix(
        "kmeans",
        X_sub,
        k,
        n_resamples=P.k_selection.bootstrap.n_resamples,
        sample_fraction=P.k_selection.bootstrap.sample_fraction,
        seed=P.seeds.algorithms,
        max_n=P.phase2.consensus_n,
        **_family_kwargs(P, "kmeans"),
    )
    consensus = consensus_clustering(
        M_co, k, threshold=P.stability.consensus.co_occurrence_threshold
    )
    plot_consensus(M_co, consensus, figure_path("phase2_consensus.pdf"))

    # Does consensus beat the best single base clusterer on the same rows?
    co_idx = stratified_subsample(y_full, P.phase2.consensus_n, seed=P.seeds.algorithms)
    X_co, y_co = X_full[co_idx], y_full[co_idx]
    base = score_run(
        run_algorithm("kmeans", X_co, k, seed=P.seeds.algorithms, **_family_kwargs(P, "kmeans")),
        X_co,
        y_co,
        internal_metrics=P.metrics.internal,
        external_metrics=P.metrics.external,
        seed=P.seeds.algorithms,
    )
    cons_run = PortfolioRun(
        algorithm="consensus",
        k=k,
        labels=consensus,
        runtime_s=float("nan"),
        n_clusters=int(len(np.unique(consensus))),
        n_noise=0,
    )
    score_run(
        cons_run,
        X_co,
        y_co,
        internal_metrics=P.metrics.internal,
        external_metrics=P.metrics.external,
        seed=P.seeds.algorithms,
    )
    summary["consensus"] = {
        "n_rows": int(len(consensus)),
        "threshold": P.stability.consensus.co_occurrence_threshold,
        "mean_co_association": float(M_co[np.triu_indices_from(M_co, k=1)].mean()),
        "comparison": performance_table([base, cons_run]),
    }

    # ---- §3.6 preferred clustering + error analysis ------------------------ #
    ranked = rank_runs(full_runs, P.phase2.primary_metric)
    preferred = ranked[0]
    print(f"[phase2] preferred: {preferred.algorithm} (k={preferred.n_clusters})", flush=True)
    ea = error_analysis(
        X_full,
        preferred.labels,
        y_true=y_full,
        n_worst=P.phase2.error_analysis.n_worst,
        sample_size=P.phase2.error_analysis.sample_size,
        seed=P.seeds.algorithms,
    )
    summary["preferred"] = {
        "algorithm": preferred.algorithm,
        "k": preferred.n_clusters,
        "primary_metric": P.phase2.primary_metric,
        "ranking": [
            {"algorithm": r.algorithm, "score": r.internal.get(P.phase2.primary_metric)}
            for r in ranked
        ],
    }
    summary["error_analysis"] = {
        "n_inspected": ea.n_inspected,
        "silhouette_threshold": ea.threshold,
        "mean_silhouette": ea.mean_silhouette,
        "frac_negative_silhouette": ea.frac_negative_silhouette,
        "per_cluster_counts": ea.per_cluster_counts,
        "fraud_rate_worst": ea.label_rate_worst,
        "fraud_rate_overall": ea.label_rate_overall,
        "fraud_enrichment": (
            ea.label_rate_worst / ea.label_rate_overall
            if ea.label_rate_overall
            else None
        ),
    }
    plot_silhouette_distribution(
        X_full, preferred.labels, figure_path("phase2_silhouette.pdf"), seed=P.seeds.algorithms
    )

    # persist the preferred full-data labelling for Phase 3 + the dashboard
    np.save(processed_path("phase2_clusters.npy").as_posix(), preferred.labels)
    for run in full_runs:
        np.save(processed_path(f"phase2_labels_{run.algorithm}.npy").as_posix(), run.labels)

    summary["runtime_s"] = round(time.perf_counter() - started, 1)
    write_json(summary, results_path("phase2", "summary.json"))
    write_tables(summary)
    print(f"[phase2] done in {summary['runtime_s']}s", flush=True)
    return summary


def refresh_tables() -> dict:
    """Re-emit the tables from the last run's persisted summary."""
    path = results_path("phase2", "summary.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the phase2 stage first")
    summary = json.loads(path.read_text())
    write_tables(summary)
    print(f"[phase2] tables refreshed from {path}", flush=True)
    return summary


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        refresh_tables()
    else:
        main()
