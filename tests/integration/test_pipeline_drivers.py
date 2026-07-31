"""End-to-end runs of both analysis pipelines on tiny synthetic artefacts.

The unit tests cover each analysis in isolation; this exercises the drivers as a
grader does — every stage in order, writing every artefact — but on ~600 rows and
a handful of epochs so it finishes in seconds. It is the test that catches
plumbing failures: a params key the driver reads but does not exist, a table
written to the wrong path, a summary that cannot be serialised to JSON.

All repo-relative roots are redirected into ``tmp_path`` so a test run never
touches committed results, figures, or the model registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_params(tmp_path: Path) -> Path:
    """Real params.yaml with every size knob shrunk to test scale."""
    raw = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    raw["algorithms"]["k_range"] = [2, 3, 4]
    raw["algorithms"]["families"]["kmeans"]["n_init"] = 2
    raw["algorithms"]["families"]["hdbscan"].update({"min_cluster_size": 15, "min_samples": 5})
    raw["k_selection"]["silhouette"]["sample_size"] = 200
    raw["k_selection"]["gap"]["n_refs"] = 2
    raw["k_selection"]["bootstrap"]["n_resamples"] = 3
    raw["metrics"]["internal"] = ["silhouette", "davies_bouldin", "calinski_harabasz", "dunn"]
    raw["stability"]["n_seeds"] = 3
    raw["phase2"].update(
        {
            "portfolio_n": 300,
            "k_selection_n": 300,
            "ablation_n": 150,
            "consensus_n": 150,
            "error_analysis": {"n_worst": 20, "sample_size": 300},
        }
    )
    raw["phase3"].update({"train_n": 300, "eval_n": 300})
    raw["deep"]["autoencoder"].update(
        {"latent_dim": 4, "hidden_dims": [16], "epochs": 5, "batch_size": 64}
    )
    raw["deep"]["ae_kmeans"]["epochs"] = 3
    raw["deep"]["dec"].update({"init_epochs": 5, "dec_epochs": 6, "update_interval": 2})
    raw["deep"]["dec_infonce"]["n_negatives"] = 32
    raw["interpretation"].update({"tree_max_depth": 3, "tree_min_samples_leaf": 10, "shap_background": 50})
    raw["significance"].update({"n_bootstrap": 30, "n_permutations": 30})
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Synthetic prepared-data artefacts, with every output root redirected to tmp_path."""
    from clustering_analysis import (
        io_utils,
        registry,
        reporting,
        run_deep_clustering,
        run_portfolio,
    )

    n_features = len(yaml.safe_load((REPO_ROOT / "params.yaml").read_text())["scaling"]["v_features"]) + 3
    rng = np.random.default_rng(0)
    n = 600
    X = np.vstack(
        [
            rng.normal(0, 1, (n // 2, n_features)),
            rng.normal(4, 1, (n // 2, n_features)),
        ]
    )
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=30, replace=False)] = 1  # rare positive class

    processed = tmp_path / "data" / "processed"
    interim = tmp_path / "data" / "interim"
    processed.mkdir(parents=True)
    interim.mkdir(parents=True)
    np.save(processed / "X_pca.npy", X)
    np.save(processed / "X_scaled.npy", X)
    np.save(processed / "umap_2d.npy", X[:, :2])
    np.save(processed / "labels.npy", y)

    monkeypatch.setattr(io_utils, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(reporting, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(reporting, "FIGURES_ROOT", tmp_path / "reports" / "figures")
    monkeypatch.setattr(run_portfolio, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_deep_clustering, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(registry, "REGISTRY_ROOT", tmp_path / "models" / "registry")
    return tmp_path, _tiny_params(tmp_path)


def test_portfolio_pipeline_writes_every_declared_artefact(sandbox):
    from clustering_analysis.run_portfolio import main

    root, params_path = sandbox
    summary = main(params_path)

    # the summary keys that the report and the deep-clustering pipeline both rely on
    for key in (
        "k_selection",
        "selected_k",
        "portfolio_full",
        "portfolio_subsample",
        "agreement",
        "linkage_comparison",
        "dendrogram_cuts",
        "gmm_covariance",
        "metric_ablation",
        "seed_stability",
        "consensus",
        "preferred",
        "error_analysis",
    ):
        assert key in summary, f"portfolio summary missing {key!r}"

    # all four families appear in the shared-subsample comparison
    assert {r["algorithm"] for r in summary["portfolio_subsample"]} == {
        "kmeans",
        "ward",
        "gmm",
        "hdbscan",
    }
    # all four linkage criteria, each with a cophenetic correlation
    assert {d["linkage"] for d in summary["linkage_comparison"]} == {
        "single",
        "complete",
        "average",
        "ward",
    }
    # at least three covariance structures, as the brief requires
    assert len(summary["gmm_covariance"]) >= 3
    # five criteria for k
    assert {m["method"] for m in summary["k_selection"]["methods"]} == {
        "elbow",
        "silhouette",
        "gap",
        "bic",
        "bootstrap",
    }

    assert (root / "results" / "phase2" / "summary.json").exists()
    assert (root / "data" / "processed" / "phase2_clusters.npy").exists()
    for figure in (
        "phase2_k_selection.pdf",
        "phase2_agreement.pdf",
        "phase2_consensus.pdf",
        "phase2_dendrograms.pdf",
        "phase2_silhouette.pdf",
    ):
        assert (root / "reports" / "figures" / figure).exists(), figure
    tables = {p.name for p in (root / "reports" / "tables").glob("*.tex")}
    assert {
        "phase2_k_selection.tex",
        "phase2_portfolio_full.tex",
        "phase2_portfolio_subsample.tex",
        "phase2_agreement.tex",
        "phase2_linkage.tex",
        "phase2_cuts.tex",
        "phase2_gmm_covariance.tex",
        "phase2_metric_ablation.tex",
        "phase2_consensus.tex",
        "phase2_error_analysis.tex",
    } <= tables


def test_portfolio_summary_is_valid_json_without_nan_tokens(sandbox):
    from clustering_analysis.run_portfolio import main

    root, params_path = sandbox
    main(params_path)
    text = (root / "results" / "phase2" / "summary.json").read_text()
    json.loads(text)  # raises if NaN leaked through as a bare token
    assert "NaN" not in text


def test_deep_pipeline_writes_every_declared_artefact(sandbox):
    from clustering_analysis.run_deep_clustering import main as deep_main
    from clustering_analysis.run_portfolio import main as portfolio_main

    root, params_path = sandbox
    portfolio_main(params_path)
    summary = deep_main(params_path)

    for key in (
        "track_comparison",
        "agreement",
        "best_deep",
        "significance",
        "profiles",
        "exemplars",
        "domain_labels",
        "surrogate_tree",
        "downstream",
        "fairness",
        "sensitivity",
        "drift",
        "registry_entry",
    ):
        assert key in summary, f"deep-clustering summary missing {key!r}"

    # the comparison must include the classical baseline and all three deep models
    algorithms = {r["algorithm"] for r in summary["track_comparison"]}
    assert {"ae_kmeans", "dec", "dec_infonce"} <= algorithms
    assert any(a.startswith("phase2:") for a in algorithms)

    assert summary["downstream"]["conditional_modelling"]["n_test"] > 0
    assert summary["downstream"]["anomaly_detection"]["n_flagged"] > 0
    assert {a["attribute"] for a in summary["fairness"]} == {
        "log_amount quartile",
        "time-of-day quartile",
    }

    assert (root / "results" / "phase3" / "summary.json").exists()
    assert (root / "models" / "registry" / "index.json").exists()
    for figure in ("phase3_deep_losses.pdf", "phase3_latent.pdf"):
        assert (root / "reports" / "figures" / figure).exists(), figure
    tables = {p.name for p in (root / "reports" / "tables").glob("*.tex")}
    assert {
        "phase3_track_comparison.tex",
        "phase3_agreement.tex",
        "phase3_significance.tex",
        "phase3_profiles.tex",
        "phase3_exemplars.tex",
        "phase3_domain_labels.tex",
        "phase3_rules.tex",
        "phase3_shap.tex",
        "phase3_downstream.tex",
        "phase3_anomaly.tex",
        "phase3_fairness.tex",
        "phase3_sensitivity.tex",
        "phase3_drift.tex",
    } <= tables


def test_deep_pipeline_refuses_to_run_before_the_portfolio_pipeline(sandbox):
    """A missing Phase 2 summary must fail with a message that says what to run."""
    from clustering_analysis.run_deep_clustering import main

    root, params_path = sandbox
    np.save(root / "data" / "processed" / "phase2_clusters.npy", np.zeros(600, dtype=int))
    with pytest.raises(FileNotFoundError, match="phase2"):
        main(params_path)


def test_deep_pipeline_registry_entry_carries_audit_metadata(sandbox):
    from clustering_analysis.run_deep_clustering import main as deep_main
    from clustering_analysis.run_portfolio import main as portfolio_main

    root, params_path = sandbox
    portfolio_main(params_path)
    deep_main(params_path)
    entries = json.loads((root / "models" / "registry" / "index.json").read_text())["entries"]
    assert len(entries) == 1
    entry = entries[0]
    for field in ("fit_date", "data_version", "metrics", "algorithm", "k"):
        assert field in entry
