"""Streamlit dashboard for interactive cluster exploration (brief §4.6).

Run with: ``make dashboard`` or ``streamlit run src/clustering_analysis/dashboard.py``

Four pages, matching the brief's minimum:

  1. **Overview** — dataset scale, feature summary, chosen algorithm and k.
  2. **Cluster explorer** — 2D UMAP coloured by cluster, filter controls, and the
     profile of the selected cluster.
  3. **Evaluation** — every internal and external metric, the algorithm-agreement
     heatmap, and the stability diagnostics.
  4. **Live assignment** — type a record or upload a CSV; the record is assigned
     to the nearest cluster and its distance to every centroid is shown.

Every page reads artefacts that ``run_portfolio`` and ``run_deep_clustering``
already persisted (``results/*/summary.json``, ``data/processed/*.npy``), so the
dashboard can never display a number the reports do not also contain. The pure
helpers below take no Streamlit dependency, which keeps them unit-testable
without a browser.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _processed(name: str) -> Path:
    return REPO_ROOT / "data" / "processed" / name


def load_summary(section: str) -> dict | None:
    """Load one pipeline's persisted result summary, or None if it has not run.

    ``section`` is the directory under ``results/``. Those directories keep the
    course's phase names (``phase2``, ``phase3``) because the reports and the
    committed results already reference them.
    """
    path = REPO_ROOT / "results" / section / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_artifacts(sample_size: int = 10000, seed: int = 0):
    """Load persisted arrays; subsample for responsive plotting."""
    rng = np.random.default_rng(seed)
    X_scaled = np.load(_processed("X_scaled.npy"))
    umap_2d = np.load(_processed("umap_2d.npy"))
    labels = np.load(_processed("labels.npy"))
    try:
        clusters = np.load(_processed("phase2_clusters.npy"))
    except FileNotFoundError:
        clusters = labels
    n = len(X_scaled)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        return X_scaled[idx], umap_2d[idx], labels[idx], clusters[idx]
    return X_scaled, umap_2d, labels, clusters


def cluster_profile_df(
    X: np.ndarray, clusters: np.ndarray, labels: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """One row per cluster: size, fraud rate, mean of each feature."""
    rows = []
    for c in np.unique(clusters):
        if c == -1:
            continue
        mask = clusters == c
        rows.append(
            {
                "cluster": int(c),
                "size": int(mask.sum()),
                "fraud_rate": float(labels[mask].mean()) if labels is not None else 0.0,
                **{fn: float(X[mask, i].mean()) for i, fn in enumerate(feature_names)},
            }
        )
    return pd.DataFrame(rows)


def cluster_centroids(X: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid per non-noise cluster. Returns ``(cluster_ids, centroids)``."""
    ids = np.array([c for c in np.unique(clusters) if c != -1])
    if len(ids) == 0:
        raise ValueError("No non-noise clusters to build centroids from")
    centroids = np.stack([X[clusters == c].mean(axis=0) for c in ids])
    return ids, centroids


def assign_to_nearest_cluster(
    record: np.ndarray, ids: np.ndarray, centroids: np.ndarray
) -> tuple[int, pd.DataFrame]:
    """Assign one record to its nearest centroid and rank all distances.

    Returns ``(assigned_cluster, distance_table)``. The full distance table
    matters as much as the assignment: a record nearly equidistant from two
    centroids is a boundary case a reviewer should treat differently from one
    sitting deep inside a cluster.
    """
    record = np.asarray(record, dtype=float).ravel()
    if record.shape[0] != centroids.shape[1]:
        raise ValueError(
            f"Record has {record.shape[0]} features, centroids have {centroids.shape[1]}"
        )
    dists = np.linalg.norm(centroids - record, axis=1)
    order = np.argsort(dists)
    table = pd.DataFrame(
        {
            "cluster": ids[order].astype(int),
            "distance": dists[order],
            "rank": np.arange(1, len(ids) + 1),
        }
    )
    return int(ids[int(np.argmin(dists))]), table


# --------------------------------------------------------------------------- #
# Streamlit pages
# --------------------------------------------------------------------------- #
def _page_overview(st, P, X_scaled, labels, clusters, portfolio, deep):
    st.header("Overview")
    full_labels = np.load(_processed("labels.npy"), mmap_mode="r")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{full_labels.shape[0]:,}")
    c2.metric("Features", f"{X_scaled.shape[1]}")
    c3.metric("Fraud rate", f"{float(np.asarray(full_labels).mean()):.3%}")
    c4.metric("Clusters", f"{len(np.unique(clusters[clusters != -1]))}")

    if portfolio:
        st.subheader("Chosen algorithm and k")
        st.write(
            f"Preferred algorithm: **{portfolio['preferred']['algorithm']}** at "
            f"**k = {portfolio['selected_k']}**, chosen on "
            f"`{portfolio['preferred']['primary_metric']}`."
        )
        votes = portfolio["k_selection"]["votes"]
        st.caption("Votes cast for each k by the five selection criteria:")
        st.bar_chart(pd.Series({int(k): v for k, v in votes.items()}, name="votes"))
    else:
        st.info(
            "Run `python -m clustering_analysis.run_portfolio` "
            "to populate the algorithm-comparison results."
        )

    if deep:
        st.write(
            f"Best deep-clustering model: **{deep['best_deep']}** "
            f"(registry entry `{deep['registry_entry']['entry_id']}`)."
        )

    st.subheader("Feature summary (scaled space)")
    names = list(P.scaling.v_features) + list(P.scaling.robust_features)
    st.dataframe(
        pd.DataFrame(
            {
                "feature": names,
                "mean": X_scaled.mean(axis=0),
                "std": X_scaled.std(axis=0),
                "min": X_scaled.min(axis=0),
                "max": X_scaled.max(axis=0),
            }
        ),
        use_container_width=True,
    )


def _page_explorer(st, P, X_scaled, umap_2d, labels, clusters):
    import matplotlib.pyplot as plt

    st.header("Cluster explorer")
    names = list(P.scaling.v_features) + list(P.scaling.robust_features)
    available = sorted(int(c) for c in np.unique(clusters))

    left, right = st.columns([2, 1])
    with right:
        colour = st.radio("Colour by", ["cluster", "Class (fraud)"])
        shown = st.multiselect("Clusters to display", available, default=available)
        only_fraud = st.checkbox("Show fraud only")
    mask = np.isin(clusters, shown)
    if only_fraud:
        mask = mask & (labels == 1)

    with left:
        fig, ax = plt.subplots(figsize=(7, 6))
        if not mask.any():
            ax.set_title("No points match the current filters")
        elif colour == "cluster":
            sc = ax.scatter(
                umap_2d[mask, 0], umap_2d[mask, 1], c=clusters[mask], s=4, cmap="tab20", alpha=0.5
            )
            fig.colorbar(sc, label="cluster")
            ax.set_title("UMAP 2D — cluster assignment")
        else:
            fraud = labels[mask] == 1
            pts = umap_2d[mask]
            ax.scatter(pts[~fraud, 0], pts[~fraud, 1], s=2, alpha=0.2, label="legitimate")
            ax.scatter(pts[fraud, 0], pts[fraud, 1], s=12, color="red", label="fraud")
            ax.legend()
            ax.set_title("UMAP 2D — Class overlay")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        st.pyplot(fig)

    st.subheader("Selected cluster profile")
    choices = [c for c in available if c != -1] or available
    selected = st.selectbox("Cluster", choices)
    profiles = cluster_profile_df(X_scaled, clusters, labels, names)
    if not profiles.empty and selected in set(profiles["cluster"]):
        row = profiles[profiles["cluster"] == selected].iloc[0]
        overall = float(labels.mean())
        c1, c2, c3 = st.columns(3)
        c1.metric("Size", f"{int(row['size']):,}")
        c2.metric("Fraud rate", f"{row['fraud_rate']:.3%}")
        c3.metric(
            "Enrichment vs population",
            f"{row['fraud_rate'] / overall:.1f}x" if overall else "n/a",
        )
        deviations = pd.Series({fn: row[fn] for fn in names}).sort_values(
            key=np.abs, ascending=False
        )
        st.caption("Most distinctive features (cluster mean, scaled units):")
        st.bar_chart(deviations.head(10))
    st.dataframe(profiles, use_container_width=True)


def _page_evaluation(st, portfolio, deep):
    import matplotlib.pyplot as plt

    st.header("Evaluation")
    if not portfolio:
        st.info(
            "Run `python -m clustering_analysis.run_portfolio` to populate this page."
        )
        return

    st.subheader("Quality measures on the full dataset")
    st.dataframe(pd.DataFrame(portfolio["portfolio_full"]), use_container_width=True)
    st.subheader("All four algorithm families, on identical rows")
    st.dataframe(pd.DataFrame(portfolio["portfolio_subsample"]), use_container_width=True)

    st.subheader("Algorithm agreement (pairwise ARI)")
    names = portfolio["agreement"]["names"]
    M = np.array(portfolio["agreement"]["matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M, cmap="viridis", vmin=-0.05, vmax=1.0)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
    fig.colorbar(im, label="ARI")
    st.pyplot(fig)

    st.subheader("Stability diagnostics")
    st.dataframe(
        pd.DataFrame(
            [{"algorithm": name, **vals} for name, vals in portfolio["seed_stability"].items()]
        ),
        use_container_width=True,
    )
    st.write(
        f"Mean co-association across pairs: "
        f"**{portfolio['consensus']['mean_co_association']:.3f}** "
        f"(link threshold {portfolio['consensus']['threshold']})."
    )
    st.dataframe(pd.DataFrame(portfolio["consensus"]["comparison"]), use_container_width=True)

    st.subheader("Determining k")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "method": m["method"],
                    "recommended_k": m["recommended_k"],
                    "direction": "max" if m["higher_is_better"] else "min",
                }
                for m in portfolio["k_selection"]["methods"]
            ]
        ),
        use_container_width=True,
    )

    if deep:
        st.subheader("Deep models vs the classical winner")
        st.dataframe(pd.DataFrame(deep["track_comparison"]), use_container_width=True)
        st.subheader("Significance — bootstrap CI + permutation p-value")
        st.dataframe(
            pd.DataFrame(deep["significance"]["per_clustering"]), use_container_width=True
        )


def _page_live_assignment(st, P, X_scaled, clusters):
    st.header("Live assignment")
    names = list(P.scaling.v_features) + list(P.scaling.robust_features)
    try:
        ids, centroids = cluster_centroids(X_scaled, clusters)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(
        "Records are expected in the scaled feature space produced by Phase 1 — "
        f"{len(names)} columns: {', '.join(names[:5])}, ..."
    )

    uploaded = st.file_uploader("Upload a CSV of scaled records", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in names if c not in df.columns]
        if missing:
            st.error(f"CSV is missing {len(missing)} expected columns, e.g. {missing[:5]}")
            return
        assigned, nearest = [], []
        for _, row in df[names].iterrows():
            cluster, table = assign_to_nearest_cluster(row.to_numpy(), ids, centroids)
            assigned.append(cluster)
            nearest.append(float(table["distance"].iloc[0]))
        out = df.copy()
        out["assigned_cluster"] = assigned
        out["distance_to_centroid"] = nearest
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download assignments", out.to_csv(index=False), "assignments.csv", "text/csv"
        )
        return

    st.write("Or enter a single record:")
    medians = np.median(X_scaled, axis=0)
    values = []
    cols = st.columns(4)
    for i, name in enumerate(names):
        with cols[i % 4]:
            values.append(
                st.number_input(name, value=float(medians[i]), format="%.4f", key=f"live_{name}")
            )
    if st.button("Assign to nearest cluster"):
        cluster, table = assign_to_nearest_cluster(np.array(values), ids, centroids)
        st.success(f"Nearest cluster: **{cluster}**")
        st.dataframe(table, use_container_width=True)
        st.bar_chart(table.set_index("cluster")["distance"])


def main():
    import streamlit as st

    from clustering_analysis.config import load_params

    P = load_params(REPO_ROOT / "params.yaml")
    st.set_page_config(page_title="DS-10 Cluster Explorer", layout="wide")
    st.title("DS-10 Credit Card Fraud — Cluster Explorer")

    X_scaled, umap_2d, labels, clusters = load_artifacts(
        sample_size=P.dashboard.sample_size, seed=P.seeds.global_
    )
    # Section names under results/ mirror the course phases; the local names
    # describe what each summary actually contains.
    portfolio = load_summary("phase2")
    deep = load_summary("phase3")

    page = st.sidebar.radio(
        "Page", ["Overview", "Cluster explorer", "Evaluation", "Live assignment"]
    )
    st.sidebar.caption(
        f"Plots use a {len(X_scaled):,} row sample (seed {P.seeds.global_}) to stay responsive."
    )

    if page == "Overview":
        _page_overview(st, P, X_scaled, labels, clusters, portfolio, deep)
    elif page == "Cluster explorer":
        _page_explorer(st, P, X_scaled, umap_2d, labels, clusters)
    elif page == "Evaluation":
        _page_evaluation(st, portfolio, deep)
    else:
        _page_live_assignment(st, P, X_scaled, clusters)


if __name__ == "__main__":
    main()
