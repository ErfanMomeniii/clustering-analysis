"""Verify that numbers quoted in report prose match the pipeline's own results.

Tables are generated, so they cannot drift. Prose is written by hand, and the
brief is explicit that claims which cannot be traced to a reproducible
experiment will be queried. This module closes that gap: each check pairs a
number quoted in a report with the path in ``results/*/summary.json`` it came
from, so re-running a phase and getting different numbers turns into a failing
test rather than a silently stale sentence.

Skipped when a phase has not been run, so a fresh clone is not blocked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _summary(phase: str) -> dict:
    path = REPO_ROOT / "results" / phase / "summary.json"
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO_ROOT)} not present — run the {phase} stage")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def phase2() -> dict:
    return _summary("phase2")


@pytest.fixture(scope="module")
def phase3() -> dict:
    return _summary("phase3")


def _report_text(name: str) -> str:
    """Report source with ``\\num{...}`` unwrapped so checks match the bare value.

    The reports wrap figures in ``\\num{}`` for consistent typesetting; the
    assertions here care about the value, not the macro.
    """
    text = (REPO_ROOT / "reports" / name).read_text()
    text = re.sub(r"\\num\{([^}]*)\}", r"\1", text)
    # LaTeX source wraps freely, so collapse whitespace before substring checks
    return re.sub(r"\s+", " ", text)


@pytest.fixture(scope="module")
def phase2_text() -> str:
    return _report_text("phase2_report.tex")


@pytest.fixture(scope="module")
def phase3_text() -> str:
    return _report_text("phase3_report.tex")


def _by_algorithm(rows: list[dict]) -> dict[str, dict]:
    return {r["algorithm"]: r for r in rows}


def _quoted(text: str, value: float, *, decimals: int) -> bool:
    """Is ``value`` quoted in the prose at the given precision?"""
    return f"{value:.{decimals}f}" in text


# --- Phase 2 ---------------------------------------------------------------- #
def test_selected_k_matches_prose(phase2, phase2_text):
    assert f"$k={phase2['selected_k']}$" in phase2_text


def test_per_criterion_k_recommendations_match_prose(phase2, phase2_text):
    by_method = {m["method"]: m["recommended_k"] for m in phase2["k_selection"]["methods"]}
    assert f"Elbow\nselects $k={by_method['elbow']}$" in phase2_text or (
        f"selects $k={by_method['elbow']}$" in phase2_text
    )
    for method in ("silhouette", "gap"):
        assert f"$k={by_method[method]}$" in phase2_text
    assert f"select $k={by_method['bic']}$" in phase2_text


def test_hdbscan_cluster_and_noise_counts_match_prose(phase2, phase2_text):
    hdbscan = _by_algorithm(phase2["portfolio_full"])["hdbscan"]
    assert f"{hdbscan['n_clusters']}" in phase2_text
    assert f"{hdbscan['n_noise']:,}".replace(",", "{,}") in phase2_text or str(hdbscan["n_noise"]) in phase2_text
    noise_pct = round(100 * hdbscan["n_noise"] / phase2["n_rows_full"])
    assert f"{noise_pct}\\% of the dataset" in phase2_text


def test_runtime_claim_is_reproducible_not_a_literal_wall_clock(phase2, phase2_text):
    """Prose must not quote a wall-clock figure: it differs on every machine.

    The claim it may make is the *ratio* between the families, which is stable.
    """
    rows = _by_algorithm(phase2["portfolio_full"])
    ratio = rows["hdbscan"]["runtime_s"] / rows["kmeans"]["runtime_s"]
    assert ratio > 100, "prose claims HDBSCAN is hundreds of times slower"
    assert "several hundred" in phase2_text
    for algorithm in ("hdbscan", "kmeans"):
        literal = f"{round(rows[algorithm]['runtime_s'])}\\,s"
        assert literal not in phase2_text, (
            f"prose quotes a per-run wall-clock ({literal}); keep it in the table instead"
        )


def test_ward_largest_cluster_share_matches_prose(phase2, phase2_text):
    ward = next(d for d in phase2["linkage_comparison"] if d["linkage"] == "ward")
    assert f"{100 * ward['largest_cluster_frac']:.1f}\\%" in phase2_text


def test_cophenetic_correlations_match_prose(phase2, phase2_text):
    by_linkage = {d["linkage"]: d["cophenetic_corr"] for d in phase2["linkage_comparison"]}
    assert _quoted(phase2_text, by_linkage["average"], decimals=3)
    assert _quoted(phase2_text, by_linkage["ward"], decimals=3)
    # the prose claims average is the highest and ward the lowest
    assert by_linkage["average"] == max(by_linkage.values())
    assert by_linkage["ward"] == min(by_linkage.values())


def test_gmm_full_covariance_wins_on_bic_as_prose_claims(phase2, phase2_text):
    rows = phase2["gmm_covariance"]
    assert rows[0]["covariance_type"] == "full", "prose says full covariance wins on BIC"
    assert rows[-1]["covariance_type"] == "tied", "prose says tied performs worst"
    by_type = {r["covariance_type"]: r for r in rows}
    assert str(by_type["full"]["n_parameters"]) in phase2_text
    assert str(by_type["diag"]["n_parameters"]) in phase2_text


def test_external_metric_claims_match_prose(phase2, phase2_text):
    rows = _by_algorithm(phase2["portfolio_full"])
    assert _quoted(phase2_text, rows["kmeans"]["ari"], decimals=4)
    assert _quoted(phase2_text, rows["gmm"]["ari"], decimals=4)
    assert _quoted(phase2_text, rows["kmeans"]["purity"], decimals=4)
    # the prose asserts every ARI is negligible
    all_ari = [r["ari"] for r in phase2["portfolio_full"] + phase2["portfolio_subsample"]]
    assert max(abs(a) for a in all_ari) < 0.01


def test_metric_ablation_claims_match_prose(phase2, phase2_text):
    by_metric = {r["metric"]: r for r in phase2["metric_ablation"]}
    assert by_metric["manhattan"]["ari_vs_reference"] == pytest.approx(1.0), (
        "prose says Manhattan reproduces the Euclidean partition exactly"
    )
    assert abs(by_metric["cosine"]["ari_vs_reference"]) < 0.01, (
        "prose says cosine produces an unrelated partition"
    )
    assert f"{by_metric['euclidean']['silhouette']:.2f}" in phase2_text


def test_seed_stability_values_match_prose(phase2, phase2_text):
    assert _quoted(phase2_text, phase2["seed_stability"]["kmeans"]["mean_pairwise_ari"], decimals=3)
    assert _quoted(phase2_text, phase2["seed_stability"]["gmm"]["mean_pairwise_ari"], decimals=3)
    assert (
        phase2["seed_stability"]["gmm"]["mean_pairwise_ari"]
        < phase2["seed_stability"]["kmeans"]["mean_pairwise_ari"]
    ), "prose says GMM is the less stable of the two"


def test_error_analysis_enrichment_is_quoted_where_it_is_used(phase2, phase3_text):
    """The Phase 2 table carries the numbers; the Phase 3 report cites them."""
    ea = phase2["error_analysis"]
    assert ea["fraud_enrichment"] > 1.0
    assert str(ea["n_inspected"]) in phase3_text
    assert f"{round(ea['fraud_enrichment'])}" in phase3_text


# --- Phase 3 ---------------------------------------------------------------- #
def test_deep_track_loses_to_baseline_as_prose_claims(phase3, phase3_text):
    rows = _by_algorithm(phase3["track_comparison"])
    baseline = next(r for name, r in rows.items() if name.startswith("phase2:"))
    best_deep = rows[phase3["best_deep"]]
    assert baseline["silhouette"] > best_deep["silhouette"], "prose says the deep track loses"
    assert _quoted(phase3_text, baseline["silhouette"], decimals=3)
    assert _quoted(phase3_text, baseline["ari"], decimals=4)
    assert _quoted(phase3_text, best_deep["silhouette"], decimals=3)


def test_head_to_head_confidence_interval_matches_prose(phase3, phase3_text):
    h2h = phase3["significance"]["best_deep_vs_phase2"]
    assert h2h["significant"] is True, "prose says the difference is significant"
    assert f"[{h2h['ci_lower']:.4f}, {h2h['ci_upper']:.4f}]" in phase3_text
    assert h2h["ci_upper"] < 0, "prose says the deep model is significantly worse"


def test_cluster_profile_shares_match_prose(phase3, phase3_text):
    profiles = {p["cluster"]: p for p in phase3["profiles"]}
    for p in profiles.values():
        assert f"{100 * p['share']:.1f}\\%" in phase3_text


def test_surrogate_tree_fidelity_matches_prose(phase3, phase3_text):
    tree = phase3["surrogate_tree"]
    assert f"{100 * tree['accuracy']:.1f}\\%" in phase3_text
    assert str(tree["n_rules"]) in phase3_text
    top = max(tree["rules"], key=lambda r: r["support"])
    assert f"{top['support']:,}".replace(",", "{,}") in phase3_text or str(top["support"]) in phase3_text
    assert _quoted(phase3_text, top["precision"], decimals=3)


def test_downstream_lift_matches_prose(phase3, phase3_text):
    cond = phase3["downstream"]["conditional_modelling"]
    assert cond["lift"] < 1.0, "prose says the clustered approach loses"
    assert _quoted(phase3_text, cond["clustered_average_precision"], decimals=3)
    assert _quoted(phase3_text, cond["global_average_precision"], decimals=3)
    assert _quoted(phase3_text, cond["lift"], decimals=2)


def test_anomaly_ranking_numbers_match_prose(phase3, phase3_text):
    a = phase3["downstream"]["anomaly_detection"]
    assert str(a["n_flagged"]) in phase3_text
    assert f"{100 * a['recall_at_k']:.1f}\\%" in phase3_text
    assert _quoted(phase3_text, a["precision_at_k"], decimals=3)
    assert _quoted(phase3_text, a["roc_auc"], decimals=3)
    assert _quoted(phase3_text, a["average_precision"], decimals=3)
    assert _quoted(phase3_text, a["enrichment"], decimals=1)


def test_fairness_ratios_match_prose(phase3, phase3_text):
    by_attribute = {a["attribute"]: a for a in phase3["fairness"]}
    for audit in by_attribute.values():
        assert _quoted(phase3_text, audit["max_representation_ratio"], decimals=2)
        assert audit["concentrated_clusters"] == [], "prose says no cluster trips the threshold"


def test_sensitivity_values_match_prose(phase3, phase3_text):
    by_variant = {s["variant"]: s for s in phase3["sensitivity"]}
    for s in by_variant.values():
        assert abs(s["ari_vs_reference"]) < 0.1, "prose says both variants show near-zero agreement"
    assert _quoted(phase3_text, by_variant["pca_space"]["ari_vs_reference"], decimals=4)


def test_drift_findings_match_prose(phase3, phase3_text):
    drift = phase3["drift"]
    assert drift["summary"]["refit_required"] is True
    assert f"{drift['summary']['n_drifted']} of {drift['summary']['n_features']}" in phase3_text
    top = drift["top_features"][:3]
    for feature in top:
        assert _quoted(phase3_text, feature["psi"], decimals=2)
    assert drift["split_axis"].startswith("raw Time"), (
        "prose states the split uses the raw monotonic Time column"
    )


def test_registry_entry_is_recorded(phase3):
    assert phase3["registry_entry"]["entry_id"]
    index = REPO_ROOT / "models" / "registry" / "index.json"
    if index.exists():
        ids = {e["entry_id"] for e in json.loads(index.read_text())["entries"]}
        assert phase3["registry_entry"]["entry_id"] in ids
