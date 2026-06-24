import numpy as np
import pandas as pd
import pytest
from clustering_analysis.features import engineer_features, FeatureEngineer

def _interim(amount=10.0, time=0.0):
    return pd.DataFrame(
        [{**{f"V{i}": 0.0 for i in range(1, 29)}, "Time": time, "Amount": amount, "Class": 0}]
    )

def test_engineer_features_adds_log_amount():
    df = _interim(amount=10.0)
    out = engineer_features(df, period_seconds=86400)
    assert "log_amount" in out.columns
    assert out["log_amount"].iloc[0] == pytest.approx(np.log1p(10.0))

def test_engineer_features_adds_cyclic_time():
    df = _interim(time=0.0)
    out = engineer_features(df, period_seconds=86400)
    assert out["time_sin"].iloc[0] == pytest.approx(0.0)
    assert out["time_cos"].iloc[0] == pytest.approx(1.0)

def test_engineer_features_cyclic_at_quarter_period():
    df = _interim(time=86400 / 4)
    out = engineer_features(df, period_seconds=86400)
    assert out["time_sin"].iloc[0] == pytest.approx(1.0)
    assert out["time_cos"].iloc[0] == pytest.approx(0.0, abs=1e-9)

def test_engineer_features_drops_raw_time_amount_class():
    df = _interim()
    out = engineer_features(df, period_seconds=86400)
    assert "Time" not in out.columns
    assert "Amount" not in out.columns
    assert "Class" not in out.columns

def test_engineer_features_preserves_v_columns():
    df = _interim()
    out = engineer_features(df, period_seconds=86400)
    for i in range(1, 29):
        assert f"V{i}" in out.columns

def test_engineer_features_idempotent_on_columns():
    df = _interim()
    out1 = engineer_features(df, period_seconds=86400)
    out2 = engineer_features(df, period_seconds=86400)
    pd.testing.assert_frame_equal(out1, out2)

def test_feature_engineer_transformer_is_sklearn_compatible():
    df = _interim()
    fe = FeatureEngineer(period_seconds=86400)
    out = fe.fit_transform(df.drop(columns=["Class"]))
    assert "log_amount" in out.columns
    assert "time_sin" in out.columns
    assert fe.get_feature_names_out().tolist() == out.columns.tolist()
