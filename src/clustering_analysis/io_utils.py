from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def raw_path(name: str) -> Path:
    return DATA_ROOT / "raw" / name


def interim_path(name: str) -> Path:
    return DATA_ROOT / "interim" / name


def processed_path(name: str) -> Path:
    return DATA_ROOT / "processed" / name


def ensure_parents(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_parents(path)
    df.to_parquet(path, index=False)
    return path
