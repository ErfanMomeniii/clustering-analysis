"""Comparative evaluation of a clustering portfolio (Phase 2 §3.6).

Turns a set of fitted labellings into the three artefacts the brief's
comparative analysis requires:

  - a **performance table**: one row per (algorithm, k) with every internal and
    external metric plus wall-clock runtime;
  - **algorithm-pair agreement**: the pairwise ARI matrix between final
    labellings, where disagreement is often the most informative result;
  - an **error analysis** of the preferred clustering: the lowest-silhouette
    points, characterised so they can be called noise, boundary, or a missed
    subcluster rather than merely counted.

A distance-metric ablation is included here too, since comparing a clustering
under two metrics (brief §2.6) is an evaluation question, not an algorithm one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from .algorithms import fit_algorithm
from .metrics import higher_is_better, score_external, score_internal


@dataclass
class PortfolioRun:
    """One fitted (algorithm, k) result plus its timing and metric scores."""

    algorithm: str
    k: int
    labels: np.ndarray
    runtime_s: float
    n_clusters: int
    n_noise: int
    internal: dict[str, float] = field(default_factory=dict)
    external: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "k_requested": self.k,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "runtime_s": round(self.runtime_s, 3),
            **{k: v for k, v in self.internal.items()},
            **{k: v for k, v in self.external.items()},
        }


def run_algorithm(
    name: str,
    X: np.ndarray,
    k: int,
    *,
    seed: int,
    **fit_kwargs,
) -> PortfolioRun:
    """Fit one algorithm, timing it, and record cluster/noise counts."""
    start = time.perf_counter()
    labels = fit_algorithm(name, X, k, seed=seed, **fit_kwargs)
    runtime = time.perf_counter() - start
    unique = np.unique(labels)
    return PortfolioRun(
        algorithm=name,
        k=k,
        labels=labels,
        runtime_s=runtime,
        n_clusters=int(len(unique[unique != -1])),
        n_noise=int((labels == -1).sum()),
    )


def score_run(
    run: PortfolioRun,
    X: np.ndarray,
    y_true: np.ndarray | None,
    *,
    internal_metrics: list[str],
    external_metrics: list[str],
    silhouette_sample_size: int | None = None,
    seed: int = 0,
) -> PortfolioRun:
    """Attach internal (and, if labels exist, external) metric scores in place.

    A metric that is undefined for a degenerate labelling (one cluster, or
    all-noise from HDBSCAN) records NaN rather than aborting the sweep, so one
    bad configuration cannot destroy the comparison table.
    """
    for m in internal_metrics:
        try:
            run.internal[m] = score_internal(
                m, X, run.labels, sample_size=silhouette_sample_size, seed=seed
            )
        except (ValueError, KeyError):
            run.internal[m] = float("nan")
    if y_true is not None:
        for m in external_metrics:
            try:
                run.external[m] = score_external(m, y_true, run.labels)
            except (ValueError, KeyError):
                run.external[m] = float("nan")
    return run


def performance_table(runs: list[PortfolioRun]) -> list[dict]:
    """One row per run, ready for a DataFrame or a booktabs LaTeX table."""
    return [r.as_row() for r in runs]


def agreement_matrix(runs: list[PortfolioRun]) -> tuple[list[str], np.ndarray]:
    """Pairwise ARI between every pair of labellings.

    Returns ``(names, matrix)`` with the diagonal set to 1.0. Noise points are
    left in place: whether one algorithm's noise is another's cluster is part of
    the disagreement being measured.
    """
    names = [f"{r.algorithm}(k={r.n_clusters})" for r in runs]
    n = len(runs)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(runs[i].labels, runs[j].labels)
            M[i, j] = M[j, i] = ari
    return names, M


def rank_runs(runs: list[PortfolioRun], metric: str) -> list[PortfolioRun]:
    """Sort runs best-first on one metric, honouring its direction convention.

    NaN scores sort last regardless of direction, so degenerate runs never win.
    """

    def key(r: PortfolioRun):
        v = {**r.internal, **r.external}.get(metric, float("nan"))
        if np.isnan(v):
            return (1, 0.0)
        return (0, -v if higher_is_better(metric) else v)

    return sorted(runs, key=key)


@dataclass(frozen=True)
class ErrorAnalysis:
    """Characterisation of the worst-fitting points in a preferred clustering."""

    n_inspected: int
    threshold: float
    mean_silhouette: float
    frac_negative_silhouette: float
    per_cluster_counts: dict[int, int]
    worst_indices: np.ndarray
    label_rate_worst: float | None
    label_rate_overall: float | None


def error_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
    n_worst: int = 200,
    sample_size: int | None = 20000,
    seed: int = 0,
) -> ErrorAnalysis:
    """Inspect the lowest-silhouette points of a clustering (brief §3.6).

    Per-point silhouette is O(n²) in time, so it is computed on a subsample when
    ``sample_size`` is set; indices returned are positions in the original
    array. ``label_rate_worst`` vs ``label_rate_overall`` answers the question
    the brief poses — are the badly-fitted points genuinely different (here:
    enriched for fraud), or just boundary noise?
    """
    mask = labels != -1
    idx_pool = np.where(mask)[0]
    if sample_size is not None and len(idx_pool) > sample_size:
        rng = np.random.default_rng(seed)
        idx_pool = np.sort(rng.choice(idx_pool, size=sample_size, replace=False))

    sub_labels = labels[idx_pool]
    if len(np.unique(sub_labels)) < 2:
        raise ValueError("Error analysis needs >= 2 clusters in the inspected sample")

    sil = silhouette_samples(X[idx_pool], sub_labels)
    order = np.argsort(sil)
    worst_local = order[: min(n_worst, len(order))]
    worst_idx = idx_pool[worst_local]

    per_cluster = {
        int(c): int((sub_labels[worst_local] == c).sum()) for c in np.unique(sub_labels[worst_local])
    }
    return ErrorAnalysis(
        n_inspected=len(worst_idx),
        threshold=float(sil[worst_local].max()) if len(worst_local) else float("nan"),
        mean_silhouette=float(sil.mean()),
        frac_negative_silhouette=float((sil < 0).mean()),
        per_cluster_counts=per_cluster,
        worst_indices=worst_idx,
        label_rate_worst=float(np.mean(y_true[worst_idx])) if y_true is not None else None,
        label_rate_overall=float(np.mean(y_true)) if y_true is not None else None,
    )


def metric_ablation(
    X: np.ndarray,
    k: int,
    *,
    metrics: tuple[str, ...] = ("euclidean", "manhattan", "cosine"),
    linkage: str = "average",
    reference_metric: str = "euclidean",
) -> list[dict]:
    """Re-cluster under alternative distance metrics and report ARI drift.

    Answers brief §2.6's requirement to compare the clustering outcome under at
    least two metrics. Average linkage is used because it accepts an arbitrary
    metric (Ward is Euclidean-only), so the metric is the sole variable.
    ``ari_vs_reference`` near 1.0 means the metric choice is not driving the
    result; far from 1.0 means it is, and the choice needs a written defence.
    """
    labels_by_metric: dict[str, np.ndarray] = {}
    for m in metrics:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=m)
        labels_by_metric[m] = model.fit_predict(X).astype(int)

    ref = labels_by_metric[reference_metric]
    rows = []
    for m, labels in labels_by_metric.items():
        try:
            sil = score_internal("silhouette", X, labels)
        except ValueError:
            sil = float("nan")
        rows.append(
            {
                "metric": m,
                "linkage": linkage,
                "n_clusters": int(len(np.unique(labels))),
                "silhouette": sil,
                "ari_vs_reference": float(adjusted_rand_score(ref, labels)),
            }
        )
    return rows
