from clustering_analysis.config import load_params

def test_load_params_returns_typed_settings(repo_root):
    params = load_params(repo_root / "params.yaml")
    assert params.seeds.global_ == 42
    assert params.ingest.expected_rows == 284807
    assert params.features.amount_transform == "log1p"
    assert "V14" in params.scaling.v_features
    assert params.tendency.hopkins.pass_threshold == 0.6
