"""Scaling: hybrid ColumnTransformer (Strategy C) + comparison vs uniform Standard / Robust."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler


def build_scaler(strategy: str, *, v_features: list[str], robust_features: list[str]):
    if strategy == "standard":
        return ColumnTransformer(
            [("std", StandardScaler(), v_features + robust_features)],
            remainder="drop",
        )
    if strategy == "robust":
        return ColumnTransformer(
            [("rob", RobustScaler(), v_features + robust_features)],
            remainder="drop",
        )
    if strategy == "hybrid":
        return ColumnTransformer(
            [
                ("std", StandardScaler(), v_features),
                ("rob", RobustScaler(), robust_features),
            ],
            remainder="drop",
        )
    raise ValueError(f"Unknown scaling strategy: {strategy!r}")


def _summary_for(df_after: pd.DataFrame) -> dict:
    return {
        "log_amount_iqr_after": float(
            np.subtract(*np.percentile(df_after["log_amount"], [75, 25]))
        ),
        "v1_mean_after": float(df_after["V1"].mean()),
        "v1_std_after": float(df_after["V1"].std()),
    }


def fit_and_describe(
    df: pd.DataFrame, *, v_features: list[str], robust_features: list[str]
) -> dict:
    cols = v_features + robust_features
    out = {}
    for strategy in ("standard", "robust", "hybrid"):
        scaler = build_scaler(strategy, v_features=v_features, robust_features=robust_features)
        arr = scaler.fit_transform(df)
        out[strategy] = _summary_for(pd.DataFrame(arr, columns=cols))
    return out
