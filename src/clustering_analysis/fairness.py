"""Fairness and sensitivity audit (Phase 3 §4.4).

DS-10 carries no demographic attributes — the features are anonymised PCA
components — so a literal protected-attribute audit is impossible. Claiming one
anyway would be worse than skipping it. What the data does carry is
**transaction amount** and **time of day**, both of which are proxies with real
consequences: a clustering that isolates small-value or night-time transactions
would push a fraud review queue toward a particular kind of customer regardless
of actual risk. Those are the axes audited here.

The sensitivity half is dataset-agnostic: re-run the clustering under an
alternative scaler / reduction / distance metric and report the ARI against the
headline clustering. Low ARI means the clusters are an artefact of a
preprocessing choice rather than a property of the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class StratumAudit:
    """Composition of each cluster along one stratifying attribute."""

    attribute: str
    n_strata: int
    per_cluster: list[dict]
    max_representation_ratio: float
    concentrated_clusters: list[int]


def quantile_strata(values: np.ndarray, *, n_strata: int = 4) -> np.ndarray:
    """Bin a continuous attribute into quantile strata.

    Duplicate quantile edges (common for heavily tied values) collapse into
    fewer strata rather than raising — the audit reports how many it actually
    got.
    """
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_strata + 1)))
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def audit_cluster_composition(
    clusters: np.ndarray,
    strata: np.ndarray,
    *,
    attribute: str,
    concentration_threshold: float = 2.0,
) -> StratumAudit:
    """Compare each cluster's stratum mix against the population mix.

    ``representation_ratio`` is a cluster's share of a stratum divided by that
    stratum's population share. A ratio of 3 means the stratum is three times
    over-represented in the cluster. Clusters exceeding
    ``concentration_threshold`` are flagged for discussion, not automatically
    condemned — over-representation can be the finding rather than the bug.
    """
    strata_ids = np.unique(strata)
    population = {int(s): float((strata == s).mean()) for s in strata_ids}

    rows = []
    flagged: list[int] = []
    worst = 0.0
    for c in np.unique(clusters):
        mask = clusters == c
        row = {"cluster": int(c), "size": int(mask.sum())}
        ratios = []
        for s in strata_ids:
            share = float((strata[mask] == s).mean())
            ratio = share / population[int(s)] if population[int(s)] > 0 else float("nan")
            row[f"share_s{int(s)}"] = share
            row[f"ratio_s{int(s)}"] = ratio
            if np.isfinite(ratio):
                ratios.append(ratio)
        peak = max(ratios) if ratios else float("nan")
        row["max_ratio"] = peak
        rows.append(row)
        if np.isfinite(peak):
            worst = max(worst, peak)
            if peak >= concentration_threshold:
                flagged.append(int(c))

    return StratumAudit(
        attribute=attribute,
        n_strata=int(len(strata_ids)),
        per_cluster=rows,
        max_representation_ratio=float(worst),
        concentrated_clusters=flagged,
    )


@dataclass(frozen=True)
class SensitivityResult:
    variant: str
    n_clusters: int
    ari_vs_reference: float
    description: str


def preprocessing_sensitivity(
    variants: dict[str, tuple[np.ndarray, str]],
    reference_labels: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 10,
) -> list[SensitivityResult]:
    """Re-cluster each preprocessing variant and report ARI drift.

    ``variants`` maps a name to ``(feature_matrix, description)``. Every variant
    is clustered with the same algorithm and k as the reference so the
    preprocessing is the only thing that changes.
    """
    from sklearn.cluster import KMeans

    results = []
    for name, (X_variant, description) in variants.items():
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(X_variant)
        results.append(
            SensitivityResult(
                variant=name,
                n_clusters=int(len(np.unique(labels))),
                ari_vs_reference=float(adjusted_rand_score(reference_labels, labels)),
                description=description,
            )
        )
    return results
