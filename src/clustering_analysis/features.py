"""Feature engineering for DS-10: log_amount + cyclic time."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

V_COLS = [f"V{i}" for i in range(1, 29)]


def engineer_features(df: pd.DataFrame, *, period_seconds: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in V_COLS:
        out[c] = df[c].astype(float)
    out["log_amount"] = np.log1p(df["Amount"].astype(float))
    omega = 2.0 * np.pi / period_seconds
    out["time_sin"] = np.sin(omega * df["Time"].astype(float))
    out["time_cos"] = np.cos(omega * df["Time"].astype(float))
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper for engineer_features."""

    def __init__(self, period_seconds: int = 86400):
        self.period_seconds = period_seconds

    def fit(self, X, y=None):
        self._feature_names_out_ = V_COLS + ["log_amount", "time_sin", "time_cos"]
        return self

    def transform(self, X):
        return engineer_features(X, period_seconds=self.period_seconds)

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names_out_)
