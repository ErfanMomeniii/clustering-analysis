import numpy as np
import pandas as pd
import pytest

from clustering_analysis.scaling import build_scaler, fit_and_describe


def _processed_frame(n=50):
    rng = np.random.default_rng(0)
    data = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["log_amount"] = rng.normal(loc=3, scale=2, size=n)
    data["time_sin"] = rng.uniform(-1, 1, size=n)
    data["time_cos"] = rng.uniform(-1, 1, size=n)
    return pd.DataFrame(data)


def test_build_scaler_hybrid_returns_column_transformer():
    s = build_scaler(
        "hybrid",
        v_features=[f"V{i}" for i in range(1, 29)],
        robust_features=["log_amount", "time_sin", "time_cos"],
    )
    assert s.__class__.__name__ == "ColumnTransformer"


def test_build_scaler_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_scaler("bogus", v_features=[], robust_features=[])


def test_hybrid_scaler_output_shape_preserved():
    df = _processed_frame()
    s = build_scaler(
        "hybrid",
        v_features=[f"V{i}" for i in range(1, 29)],
        robust_features=["log_amount", "time_sin", "time_cos"],
    )
    out = s.fit_transform(df)
    assert out.shape == (50, 31)


def test_hybrid_scaler_v_features_standardised():
    df = _processed_frame()
    s = build_scaler(
        "hybrid",
        v_features=[f"V{i}" for i in range(1, 29)],
        robust_features=["log_amount", "time_sin", "time_cos"],
    )
    out = s.fit_transform(df)
    v_block = out[:, :28]
    assert abs(v_block.mean(axis=0)).max() < 1e-9
    assert pytest.approx(v_block.std(axis=0), abs=0.05) == np.ones(28)


def test_fit_and_describe_returns_per_strategy_summary():
    df = _processed_frame()
    desc = fit_and_describe(
        df,
        v_features=[f"V{i}" for i in range(1, 29)],
        robust_features=["log_amount", "time_sin", "time_cos"],
    )
    assert set(desc.keys()) == {"standard", "robust", "hybrid"}
    for strategy_summary in desc.values():
        assert "log_amount_iqr_after" in strategy_summary
        assert "v1_mean_after" in strategy_summary
